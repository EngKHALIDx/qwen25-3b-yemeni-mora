# تقرير تحقق حزمة التشغيل والتدريب

## النطاق

تحتوي الحزمة على إعادة تنفيذ GL-log-MoRA فوق MoRA/PEFT، ومسار تدريب Qwen2.5-3B-Instruct، ودفتر Google Colab من التنزيل إلى الدردشة، ومشغل طرفية مستقل بعد التدريب.

## ما تم التحقق منه

| الفحص | النتيجة |
|---|---|
| فحص syntax لطبقات PEFT و`train.py` و`chat_with_adapter.py` | نجح |
| صحة JSON لدفتر `colab_train_qwen25_3b_mora.ipynb` | نجحت |
| اختبار forward وQ داخل MoRA | نجح |
| اختبار merge/unmerge والتدرجات | نجح |
| اختبار التكامل والحفظ وإعادة تحميل Q | نجح |
| اختبار `GLLogMoRATrainer` والمنتظم وفصل معدل تعلم Q | نجح |
| مجموع اختبارات pytest | `4 passed in 16.01s` |

## محتوى دفتر Colab

ينفذ الدفتر استنساخ المستودع، تثبيت الاعتماديات، تثبيت PEFT المحلي، ربط Drive اختياريًا، تنزيل ملف البيانات من Google Drive، فحص SHA-256 والبنية وعدد السجلات، smoke test، التدريب الكامل، الاستئناف من checkpoint، إعادة تحميل adapter، ودردشة مباشرة مع النموذج المدرب. كما يضغط الـadapter ويسمح بنقله عند فشل ربط Drive.

## التشغيل السريع

```bash
pip install -r requirements.txt
pip install -e ./peft-mora
python train.py --config configs/qwen25_3b_mora.json --max_train_samples 128 --max_steps 5 --output_dir ./smoke_out
python chat_with_adapter.py --adapter-dir ./smoke_out --prompt "اكتب سؤالك هنا"
```

على Google Colab، يفضل استخدام دفتر `colab_train_qwen25_3b_mora.ipynb` لأنه يجهز مسار البيانات والـGPU وحفظ النتائج تلقائيًا.

## الملفات المستبعدة عمدًا

لم تُرفع أوزان Qwen، وملف البيانات الكبير، وملفات checkpoints، والـadapters الناتجة، وأي أسرار أو رموز وصول. تُحمّل هذه الموارد وقت التشغيل أو تحفظ في Google Drive، ويمنع `.gitignore` إضافتها إلى GitHub.

## ملاحظة علمية

GL-log-MoRA هنا إعادة تنفيذ مبنية على الورقة، لأن كود المؤلف الرسمي غير منشور. نجاح الاختبارات يثبت سلامة المسارات البرمجية، وتجربة Qwen الواقعية القصيرة السابقة تثبت قابلية تشغيل النموذج والبيانات والـadapter، لكنها لا تعني إعادة إنتاج كامل لأرقام الورقة دون تدريب كامل بنفس الإعدادات.
