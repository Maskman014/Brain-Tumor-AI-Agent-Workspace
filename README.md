# Brain Tumor Detection — Web App

Multi-page Flask + PyTorch (ResNet-18) site: accounts, left-sidebar dashboard, tumor-type results with Grad-CAM heatmaps, searchable history, CSV export, printable reports, Learn and About pages.

## Run
1. Train in Colab with `Brain_Tumor_Train_v2.ipynb`, download `model_files.zip`, and extract its files into the `model/` folder:
   `brain_tumor_resnet_90.pth`, `classes.json`, `metrics.json` (the About page fills itself from these).
2. `pip install -r requirements.txt`
3. `python app.py` → open http://127.0.0.1:5000 and sign up.

## Customise
- Site name: `APP_NAME` / `APP_TAG` at the top of `app.py`.
- Data is in `neuroscan.db` and `uploads/` (private). Delete both to reset.
- Deploy: set `SECRET_KEY`; on Linux use `gunicorn app:app --workers 1 --threads 4`.

Educational project — not a medical device.
