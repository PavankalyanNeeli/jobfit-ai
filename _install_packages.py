"""Download and install packages - fixed version using urlopen instead of urlretrieve."""
import ssl
import json
import urllib.request
import subprocess
import sys
import os
from pathlib import Path

WHEEL_DIR = Path("_wheels")
WHEEL_DIR.mkdir(exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

PY_TAG = "cp310"
PLATFORM = "win_amd64"

def fetch_json(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            import time; time.sleep(2)
    return None

def get_wheel_url(package_name):
    data = fetch_json(f"https://pypi.org/pypi/{package_name}/json")
    if not data:
        return None, None
    urls = data["urls"]
    # Priority order: cp310-win_amd64, cp310-any, py3-none-any, sdist
    for u in urls:
        fn = u["filename"]
        if fn.endswith(".whl") and PY_TAG in fn and PLATFORM in fn:
            return u["url"], fn
    for u in urls:
        fn = u["filename"]
        if fn.endswith(".whl") and PY_TAG in fn:
            return u["url"], fn
    for u in urls:
        fn = u["filename"]
        if fn.endswith(".whl") and "py3-none-any" in fn:
            return u["url"], fn
    for u in urls:
        if u["filename"].endswith(".tar.gz"):
            return u["url"], u["filename"]
    return None, None

def download_file(url, dest_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{attempt}/{max_retries}] Downloading...")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                data = resp.read()
                with open(dest_path, 'wb') as f:
                    f.write(data)
            size_mb = os.path.getsize(dest_path) / (1024*1024)
            print(f"  OK: {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
            if Path(dest_path).exists():
                Path(dest_path).unlink()
            import time; time.sleep(3)
    return False

NEED = ["scikit-learn", "scipy", "threadpoolctl",
        "matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "pillow", "pyparsing",
        "seaborn", "plotly", "wordcloud", "python-dateutil", "mlflow", "shap", "numba", "llvmlite", "slicer", "cloudpickle", "tqdm"]

for pkg in NEED:
    mod = pkg.replace("-","_").replace("scikit_learn","sklearn")
    try:
        __import__(mod)
        print(f"SKIP: {pkg} (already installed)")
        continue
    except ImportError:
        pass
    
    print(f"\n--- {pkg} ---")
    url, filename = get_wheel_url(pkg)
    if not url:
        print(f"  No wheel found!")
        continue
    
    dest = WHEEL_DIR / filename
    if not dest.exists():
        print(f"  URL: ...{url[-60:]}")
        if not download_file(url, dest):
            print(f"  FAILED: {pkg}")
            continue
    else:
        print(f"  Cached: {filename}")
    
    r = subprocess.run([sys.executable, "-m", "pip", "install", str(dest)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  Installed!")
    else:
        print(f"  Install error: {r.stderr[-150:]}")

print("\n" + "="*50)
print("FINAL VERIFICATION:")
for pkg in ["numpy","pandas","scikit-learn","matplotlib","seaborn","plotly","pytest","joblib","wordcloud"]:
    mod = pkg.replace("-","_").replace("scikit_learn","sklearn")
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  OK: {pkg} == {v}")
    except ImportError:
        print(f"  MISSING: {pkg}")
