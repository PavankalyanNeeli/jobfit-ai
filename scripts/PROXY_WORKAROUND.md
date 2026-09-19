# PROXY WORKAROUND

If you hit `SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC` or `CERTIFICATE_VERIFY_FAILED` during pip installs, use the following fallback ladder.

## 1. Trust the Corporate CA
Get your company's root CA cert from IT, then:
```bash
pip config set global.cert /path/to/corp-root-ca.pem
pip install scikit-learn matplotlib joblib pandas
```

## 2. Trusted Host Bypass
Pass `--trusted-host` directly to bypass verification (use with caution):
```bash
pip install scikit-learn matplotlib joblib pandas --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## 3. Offline Wheels / Conda / Colab
If local installs are completely blocked, download wheels offline and install manually, use Conda, or move Phase B to Google Colab.
