# UMKM Business Decision Stress Test Dashboard

Dashboard Streamlit bertema gelap untuk menganalisis keputusan bisnis UMKM dengan pendekatan:

- Business Decision Pipeline
- Internal vs External Uncertainty
- Scenario Planning: Best, Base, Worst
- What-if Analysis
- One-way dan Multi-way Sensitivity Analysis
- Robust Decision Analysis
- Risk Appetite dan Risk Tolerance
- Stress Testing
- Mitigation Action Plan

## File penting

- `app.py` : aplikasi utama Streamlit
- `synthetic_umkm_data.csv` : dataset UMKM
- `requirements.txt` : dependency untuk deploy Streamlit Community Cloud
- `.streamlit/config.toml` : konfigurasi tema gelap

## Cara menjalankan lokal

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Cara deploy singkat

1. Buat repository GitHub.
2. Upload semua file dalam folder ini ke repository tersebut.
3. Buka Streamlit Community Cloud.
4. Pilih repository, branch, dan file utama `app.py`.
5. Klik Deploy.
