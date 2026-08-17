# دليل التشغيل الكامل

## التشغيل عبر Google Colab

افتح `colab_train_qwen25_3b_mora.ipynb` في Google Colab واختر GPU. شغّل الخلايا بالترتيب؛ فالدفتر ينفذ استنساخ المستودع، تثبيت الاعتماديات، تثبيت `peft-mora` المحلي، ربط Google Drive اختياريًا، تنزيل ملف البيانات من Drive، التحقق من SHA-256 وعدد السجلات، اختبار دخان، التدريب، الاستئناف، إعادة تحميل الـadapter، ثم الدردشة معه.

يُفعّل ملف `configs/qwen25_3b_mora.json` GL-log-MoRA بالقيم التالية:

```text
mora_type=6
learning_rate=2e-4
gl_log_q_lr=3e-4
gl_log_lambda=0.01
gl_log_delta=1e-3
```

ابدأ باختبار الدخان قبل التدريب الكامل:

```bash
python train.py --config configs/qwen25_3b_mora.json \
  --data_path /content/qwen25_3b_mora/qwen25_3b_semantic_colab_train.jsonl \
  --max_train_samples 128 --max_steps 5 \
  --output_dir /content/qwen25_mora_smoke_test
```

بعد نجاحه شغّل خلية التدريب الكامل في الدفتر. لا ترفع ملف البيانات أو النموذج أو checkpoint إلى GitHub؛ تُنزّل البيانات وقت التشغيل ويحفظ الـadapter محليًا أو في Google Drive.

## تشغيل محلي

ثبّت الاعتماديات ثم PEFT المحلي:

```bash
pip install -r requirements.txt
pip install -e ./peft-mora
```

شغّل التدريب باستخدام `train.py` وملف إعداد مناسب للعتاد. بعد التدريب يمكن تشغيل الدردشة من الطرفية:

```bash
python chat_with_adapter.py \
  --adapter-dir /path/to/output_adapter \
  --prompt "اكتب السؤال هنا"
```

يستخدم السكربت 4-bit تلقائيًا عند توفر CUDA، ويمكن تعطيله باستخدام `--no-4bit`. وعلى CPU يجب توقع استهلاك ذاكرة وزمن أكبر بكثير مع نموذج 3B.

## التحقق

شغّل الاختبارات من جذر المستودع:

```bash
export PYTHONPATH="$PWD/peft-mora/src:$PYTHONPATH"
python -m pytest -q tests/test_gl_log_mora.py \
  tests/test_gl_log_integration.py tests/test_gl_log_trainer.py
```

تتحقق الاختبارات من تطبيق Q داخل فضاء MoRA، والدمج وفك الدمج، والتدرجات، ومرور الإعدادات عبر `LoraConfig`، وحفظ وإعادة تحميل Q، وإضافة المنتظم إلى خسارة المدرب وفصل معدل تعلم Q.

## الوضع العلمي

هذا المستودع إعادة تنفيذ لـ GL-log-MoRA اعتمادًا على الورقة، لأن كود المؤلف الرسمي غير منشور. نجاح الاختبارات وتجربة Qwen الواقعية القصيرة يثبتان قابلية التشغيل، ولكنهما لا يثبتان إعادة إنتاج أرقام الورقة دون تدريب كامل على نفس البيانات والعتاد والإعدادات.
