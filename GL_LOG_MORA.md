# GL-log-MoRA على Qwen2.5-3B-Instruct

يضيف هذا المستودع إعادة تنفيذ محلية لـ GL-log-MoRA فوق MoRA/PEFT لتدريب `Qwen/Qwen2.5-3B-Instruct` على البيانات اليمنية القانونية. الكود الرسمي لـ GL-log-MoRA غير منشور؛ لذلك فهذا التطبيق إعادة تنفيذ هندسية مبنية على المعادلات والوصف المنشور، وليس نسخة المؤلف الرسمية.

## الفكرة التنفيذية

تُنشئ طبقة MoRA مصفوفة Q مهيأة بالهوية داخل فضاء الرتبة، ثم تطبق التحويل على مصفوفة MoRA المختنقة قبل إعادة البناء:

```text
M = B · A
M_GL = B · Q · A
```

ويضاف إلى خسارة المهمة متوسط المنتظم عبر الطبقات النشطة:

```text
R_GL = -lambda * logdet(Q^T Q + delta I)
L_total = L_task + R_GL
```

القيمة `delta` موجبة لضمان الاستقرار العددي، وتُحسب الخسارة عبر دالة `slogdet` في تنفيذ الطبقة.

## التشغيل

ثبّت نسخة PEFT المحلية من داخل مجلد المستودع:

```bash
pip install -e ./peft-mora
```

ثم شغّل smoke test على GPU:

```bash
python train.py \
  --config configs/qwen25_3b_mora.json \
  --max_train_samples 128 \
  --max_steps 5 \
  --output_dir ./gl_log_mora_smoke
```

ملف الإعداد الموصى به يفعّل:

```json
{
  "use_mora": true,
  "mora_type": 6,
  "use_gl_log_mora": true,
  "gl_log_lambda": 0.01,
  "gl_log_delta": 0.001,
  "gl_log_q_lr": 0.0003
}
```

`learning_rate` يخص معاملات MoRA الأساسية وقيمته الافتراضية `2e-4`، بينما `gl_log_q_lr` يخص معاملات Q وقيمته `3e-4`. يقوم `GLLogMoRATrainer` بإنشاء مجموعة optimizer منفصلة لـQ ويضيف منتظم GL إلى loss.

## الملفات المعدلة

تم تعديل طبقات `aqlm.py` و`awq.py` و`bnb.py` و`config.py` و`gptq.py` و`layer.py` و`model.py` داخل `peft-mora`. كما عُدّل `train.py` لتمرير إعدادات GL وإنشاء `GLLogMoRATrainer`، وأضيفت اختبارات الوحدة والتكامل والمدرب.

## حدود التحقق

نجحت اختبارات الطبقة والتكامل والمدرب على نموذج صغير في النسخة المحلية. كما نجحت تجربة حقيقية قصيرة سابقة على Qwen2.5-3B-Instruct مع البيانات الأصلية، شملت forward وbackward وتحديث Q وM وحفظ وإعادة تحميل adapter. لا ينبغي تفسير ذلك على أنه إعادة إنتاج كامل لنتائج الورقة؛ إثبات النتائج التجريبية يتطلب تدريبًا كاملًا على نفس العتاد والبيانات والإعدادات المنشورة.
