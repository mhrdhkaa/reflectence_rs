from contextlib import asynccontextmanager
import glob
import os
import re
import uuid
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
import uvicorn

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Sembunyikan peringatan georeferensi untuk file JPG
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ---------------------------------------------------------------------------
# Setup Folder I/O
# ---------------------------------------------------------------------------
OS_UPLOADS = "uploads"
OS_OUTPUTS = "outputs"
OS_PREVIEWS = "previews"
for d in (OS_UPLOADS, OS_OUTPUTS, OS_PREVIEWS, "static"):
  os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".tif", ".tiff"}
PREVIEW_MAX_SIDE = 2000

# ---------------------------------------------------------------------------
# Konfigurasi PostgreSQL / pgAdmin (Mengambil dari file .env)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "calibration_logs"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS calibration_logs (
    id                 SERIAL PRIMARY KEY,
    filename           TEXT             NOT NULL,
    target_reflectance DOUBLE PRECISION NOT NULL,
    dn_red             DOUBLE PRECISION,
    dn_green           DOUBLE PRECISION,
    dn_blue            DOUBLE PRECISION,
    mult_red           DOUBLE PRECISION,
    mult_green         DOUBLE PRECISION,
    mult_blue          DOUBLE PRECISION,
    output_path        TEXT             NOT NULL,
    created_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
ALTER TABLE calibration_logs ADD COLUMN IF NOT EXISTS band_count INTEGER;
ALTER TABLE calibration_logs ADD COLUMN IF NOT EXISTS dn_values  JSONB;
ALTER TABLE calibration_logs ADD COLUMN IF NOT EXISTS mult_values JSONB;
ALTER TABLE calibration_logs ALTER COLUMN dn_red     DROP NOT NULL;
ALTER TABLE calibration_logs ALTER COLUMN dn_green   DROP NOT NULL;
ALTER TABLE calibration_logs ALTER COLUMN dn_blue    DROP NOT NULL;
ALTER TABLE calibration_logs ALTER COLUMN mult_red   DROP NOT NULL;
ALTER TABLE calibration_logs ALTER COLUMN mult_green DROP NOT NULL;
ALTER TABLE calibration_logs ALTER COLUMN mult_blue  DROP NOT NULL;
"""


# ---------------------------------------------------------------------------
# Lifespan Event Handler
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
  """Migrasi dan koneksi database saat server dinyalakan."""
  try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(MIGRATION_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database terhubung, tabel calibration_logs siap.")
  except Exception as e:
    print(f"⚠️ Warning Database (startup): {e}")

  yield


app = FastAPI(title="Reflectance Calibration API", lifespan=lifespan)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/previews", StaticFiles(directory="previews"), name="previews")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def save_log_to_db(filename, ref, dn_values, mult_values, out_path):
  def nth(values, i):
    return values[i] if i < len(values) else None

  try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
            INSERT INTO calibration_logs
            (filename, target_reflectance,
             dn_red, dn_green, dn_blue, mult_red, mult_green, mult_blue,
             band_count, dn_values, mult_values, output_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
        (
            filename,
            ref,
            nth(dn_values, 0),
            nth(dn_values, 1),
            nth(dn_values, 2),
            nth(mult_values, 0),
            nth(mult_values, 1),
            nth(mult_values, 2),
            len(dn_values),
            Json(dn_values),
            Json(mult_values),
            out_path,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
  except Exception as e:
    print(f"⚠️ Warning Database: {e}")


def find_upload(file_id: str) -> str:
  if not re.fullmatch(r"[0-9a-f]{32}", file_id or ""):
    raise HTTPException(400, "file_id tidak valid.")
  matches = glob.glob(os.path.join(OS_UPLOADS, f"{file_id}__*"))
  if not matches:
    raise HTTPException(404, "File tidak ditemukan. Unggah ulang citra.")
  return matches[0]


def original_name(path: str) -> str:
  return os.path.basename(path).split("__", 1)[1]


def stretch_to_uint8(
    arr: np.ndarray, nodata, is_8bit: bool = False
) -> np.ndarray:
  """Ubah band apa pun jadi uint8 untuk tampilan pratinjau."""
  if is_8bit:
    return arr.astype(np.uint8)
  a = arr.astype(np.float32)
  valid = a != nodata if nodata is not None else np.ones(a.shape, bool)
  valid &= np.isfinite(a)
  if not valid.any():
    return np.zeros(a.shape, np.uint8)
  lo, hi = np.percentile(a[valid], (2, 98))
  if hi <= lo:
    hi = lo + 1
  out = np.clip((a - lo) / (hi - lo), 0, 1) * 255
  out[~valid] = 0
  return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
  index_path = os.path.join("static", "index.html")
  if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
      return f.read()
  return "<h3>File static/index.html belum ditemukan.</h3>"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
  """Simpan citra (JPG/TIFF), baca metadata, dan buat pratinjau PNG."""
  name = os.path.basename(file.filename or "upload")
  ext = os.path.splitext(name)[1].lower()
  if ext not in ALLOWED_EXT:
    raise HTTPException(
        400, "Format tidak didukung. Gunakan JPG, JPEG, TIF, atau TIFF."
    )

  safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
  file_id = uuid.uuid4().hex
  path = os.path.join(OS_UPLOADS, f"{file_id}__{safe_name}")
  with open(path, "wb") as f:
    f.write(await file.read())

  try:
    with rasterio.open(path) as src:
      width, height, count = src.width, src.height, src.count
      dtype = str(src.dtypes[0])
      nodata = src.nodata
      georef = src.crs is not None
      crs = src.crs.to_string() if src.crs else None

      scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
      pw, ph = max(1, int(width * scale)), max(1, int(height * scale))
      band_ids = [1, 2, 3] if count >= 3 else [1]
      is_8bit = dtype == "uint8"

      bands = [
          stretch_to_uint8(
              src.read(b, out_shape=(ph, pw), resampling=Resampling.average),
              nodata,
              is_8bit,
          )
          for b in band_ids
      ]
      rgb = np.dstack(bands) if len(bands) == 3 else np.dstack(bands * 3)
  except Exception as e:
    os.remove(path)
    raise HTTPException(400, f"Gagal membaca citra: {e}")

  Image.fromarray(rgb, "RGB").save(os.path.join(OS_PREVIEWS, f"{file_id}.png"))

  return {
      "file_id": file_id,
      "filename": safe_name,
      "width": width,
      "height": height,
      "count": count,
      "dtype": dtype,
      "nodata": nodata,
      "georeferenced": georef,
      "crs": crs,
      "preview_url": f"/previews/{file_id}.png",
      "preview_width": pw,
      "preview_height": ph,
  }


@app.get("/sample")
async def sample_pixels(file_id: str, x: int, y: int, win: int = 5):
  """Rata-rata nilai DN asli tiap band pada jendela win×win di sekitar (x, y)."""
  path = find_upload(file_id)
  win = max(1, min(win, 101))
  half = win // 2
  with rasterio.open(path) as src:
    x = min(max(x, 0), src.width - 1)
    y = min(max(y, 0), src.height - 1)
    col0, row0 = max(0, x - half), max(0, y - half)
    col1, row1 = min(src.width, x + half + 1), min(src.height, y + half + 1)
    data = src.read(
        window=Window(col0, row0, col1 - col0, row1 - row0)
    ).astype(np.float64)
    if src.nodata is not None:
      data = np.where(data == src.nodata, np.nan, data)
    means = np.nanmean(data.reshape(src.count, -1), axis=1)

  if np.isnan(means).any():
    raise HTTPException(400, "Area yang diklik berisi nodata. Pilih titik lain.")
  return {"x": x, "y": y, "dn": [round(float(v), 4) for v in means]}


@app.get("/sample_calibrated")
async def sample_calibrated(output_filename: str, x: int, y: int, win: int = 5):
  """Membaca nilai Reflektansi (0.0 - 1.0) dari citra GeoTIFF terkalibrasi."""
  safe_name = os.path.basename(output_filename)
  path = os.path.join(OS_OUTPUTS, safe_name)
  if not os.path.exists(path):
    raise HTTPException(404, "File hasil kalibrasi tidak ditemukan.")

  win = max(1, min(win, 101))
  half = win // 2
  with rasterio.open(path) as src:
    x = min(max(x, 0), src.width - 1)
    y = min(max(y, 0), src.height - 1)
    col0, row0 = max(0, x - half), max(0, y - half)
    col1, row1 = min(src.width, x + half + 1), min(src.height, y + half + 1)
    data = src.read(
        window=Window(col0, row0, col1 - col0, row1 - row0)
    ).astype(np.float64)
    if src.nodata is not None:
      data = np.where(data == src.nodata, np.nan, data)
    means = np.nanmean(data.reshape(src.count, -1), axis=1)

  if np.isnan(means).any():
    raise HTTPException(400, "Area yang diklik berisi nodata. Pilih titik lain.")
  return {"x": x, "y": y, "reflectance": [round(float(v), 6) for v in means]}


@app.get("/calibrated_info")
async def get_calibrated_info(output_filename: str):
  """Mendapatkan metadata, statistik, dan preview URL untuk citra terkalibrasi."""
  safe_name = os.path.basename(output_filename)
  path = os.path.join(OS_OUTPUTS, safe_name)
  if not os.path.exists(path):
    raise HTTPException(404, "File hasil kalibrasi tidak ditemukan.")

  stem = os.path.splitext(safe_name)[0]
  preview_name = f"prev_{stem}.png"
  preview_path = os.path.join(OS_PREVIEWS, preview_name)

  with rasterio.open(path) as src:
    width, height, count = src.width, src.height, src.count
    dtype = str(src.dtypes[0])
    nodata = src.nodata
    georef = src.crs is not None
    crs = src.crs.to_string() if src.crs else None

    if not os.path.exists(preview_path):
      scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
      pw, ph = max(1, int(width * scale)), max(1, int(height * scale))
      band_ids = [1, 2, 3] if count >= 3 else [1]

      bands = [
          stretch_to_uint8(
              src.read(b, out_shape=(ph, pw), resampling=Resampling.average),
              nodata,
              is_8bit=False,
          )
          for b in band_ids
      ]
      rgb = np.dstack(bands) if len(bands) == 3 else np.dstack(bands * 3)
      Image.fromarray(rgb, "RGB").save(preview_path)

    stats = []
    for b in range(1, count + 1):
      arr = src.read(b)
      valid = arr != nodata if nodata is not None else np.ones(arr.shape, bool)
      valid &= np.isfinite(arr)
      if valid.any():
        v = arr[valid]
        stats.append({
            "band": b,
            "min": round(float(np.min(v)), 4),
            "max": round(float(np.max(v)), 4),
            "mean": round(float(np.mean(v)), 4),
        })
      else:
        stats.append({"band": b, "min": 0, "max": 0, "mean": 0})

  return {
      "output_filename": safe_name,
      "width": width,
      "height": height,
      "count": count,
      "dtype": dtype,
      "georeferenced": georef,
      "crs": crs,
      "preview_url": f"/previews/{preview_name}",
      "stats": stats,
  }


@app.get("/logs")
async def get_logs():
  """Riwayat kalibrasi (50 terbaru) untuk tabel di frontend."""
  try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT * FROM calibration_logs ORDER BY created_at DESC LIMIT 50"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
  except Exception as e:
    raise HTTPException(status_code=503, detail=f"Database error: {e}")

  for r in rows:
    r["created_at"] = r["created_at"].isoformat()
  return rows


@app.post("/calibrate")
async def calibrate_image(
    file_id: str = Form(...),
    target_reflectance: float = Form(...),
    dn_values: str = Form(...),
):
  if not (0 < target_reflectance <= 1):
    raise HTTPException(400, "Reflektansi acuan harus di antara 0 dan 1.")
  try:
    dns = [float(v) for v in dn_values.split(",")]
  except ValueError:
    raise HTTPException(400, "Format dn_values tidak valid.")
  if min(dns) <= 0:
    raise HTTPException(400, "Nilai DN tiap band harus lebih dari 0.")

  input_path = find_upload(file_id)
  filename = original_name(input_path)

  multipliers = [target_reflectance / dn for dn in dns]

  stem = os.path.splitext(filename)[0]
  output_filename = f"calibrated_{stem}_{file_id[:6]}.tif"
  output_path = os.path.join(OS_OUTPUTS, output_filename)

  try:
    with rasterio.open(input_path) as src:
      if len(dns) != src.count:
        raise HTTPException(
            400,
            f"Citra punya {src.count} band, tetapi DN diberikan untuk"
            f" {len(dns)} band.",
        )

      profile = src.profile.copy()
      for key in (
          "photometric",
          "compress",
          "jpeg_quality",
          "interleave",
          "tiled",
          "blockxsize",
          "blockysize",
          "predictor",
      ):
        profile.pop(key, None)
      profile.update(dtype="float32", count=src.count, driver="GTiff")
      nodata = src.nodata

      stats = []
      with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(1, src.count + 1):
          raw = src.read(band_idx)
          out = raw.astype(np.float32) * multipliers[band_idx - 1]
          if nodata is not None:
            out[raw == nodata] = nodata
          dst.write(out, band_idx)

          valid = (
              out != nodata if nodata is not None else np.ones(out.shape, bool)
          )
          valid &= np.isfinite(out)
          if valid.any():
            v = out[valid]
            stats.append({
                "band": band_idx,
                "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4),
                "mean": round(float(np.mean(v)), 4),
            })

    cal_preview_name = f"prev_{output_filename[:-4]}.png"
    cal_preview_path = os.path.join(OS_PREVIEWS, cal_preview_name)
    with rasterio.open(output_path) as cal_src:
      scale = min(1.0, PREVIEW_MAX_SIDE / max(cal_src.width, cal_src.height))
      pw, ph = max(1, int(cal_src.width * scale)), max(
          1, int(cal_src.height * scale)
      )
      band_ids = [1, 2, 3] if cal_src.count >= 3 else [1]
      bands = [
          stretch_to_uint8(
              cal_src.read(
                  b, out_shape=(ph, pw), resampling=Resampling.average
              ),
              nodata,
              is_8bit=False,
          )
          for b in band_ids
      ]
      rgb = np.dstack(bands) if len(bands) == 3 else np.dstack(bands * 3)
      Image.fromarray(rgb, "RGB").save(cal_preview_path)

  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(500, f"Gagal memproses citra: {e}")

  save_log_to_db(
      filename,
      target_reflectance,
      dns,
      multipliers,
      output_path.replace("\\", "/"),
  )

  return {
      "status": "success",
      "message": "Kalibrasi berhasil diproses",
      "multipliers": multipliers,
      "output_filename": output_filename,
      "output_url": f"/outputs/{output_filename}",
      "calibrated_preview_url": f"/previews/{cal_preview_name}",
      "stats": stats,
  }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  print("🚀 Server berjalan di http://127.0.0.1:8000")
  uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)