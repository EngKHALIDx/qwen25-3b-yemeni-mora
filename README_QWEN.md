# تدريب Qwen2.5-3B-Instruct على البيانات القانونية اليمنية باستخدام MoRA

هذه الحزمة تكيّف مستودع MoRA الأصلي لتدريب **Qwen/Qwen2.5-3B-Instruct** على مجموعة بيانات قانونية يمنية محادثية، مع الحفاظ على نسخة `peft-mora/` المحلية دون تعديل. يعتمد التدريب على **QLoRA بكمية 4-bit NF4** لتقليل استهلاك ذاكرة GPU، وعلى `Trainer` و`Accelerate` بدلاً من DeepSpeed، وهو المسار العملي لجلسة Google Colab مجانية ببطاقة T4 أو V100. قالب المحادثة مأخوذ من tokenizer الخاص بـ Qwen عبر `apply_chat_template`، وليس من prompt ثابت خاص بنماذج LLaMA [1].

> **ملاحظة مهمة:** الناتج هو adapter فقط، وليس نسخة كاملة من أوزان Qwen. عند الاستخدام يجب تحميل النموذج الأساسي `Qwen/Qwen2.5-3B-Instruct` ثم تركيب adapter فوقه. كما أن النموذج المدرب لا يستبدل مراجعة محامٍ مرخّص ولا يبرر اختلاق مواد أو وقائع خارج النص المصدر.

## محتويات الحزمة

| المسار | الوظيفة |
|---|---|
| `train.py` | نقطة الدخول الرئيسية، وتستدعي مسار التدريب الجديد. |
| `train_qwen_mora.py` | المدرب الكامل: تحميل Qwen، قراءة JSONL/GZ، تطبيق قالب المحادثة، بناء قناع loss للمساعد، ثم التدريب والحفظ. |
| `configs/qwen25_3b_mora.json` | إعدادات موصى بها لـ Colab المجاني. |
| `qwen25_3b_semantic_colab.jsonl.gz` | ملف البيانات القانوني النهائي المضمن في الحزمة. |
| `requirements.txt` | اعتماديات التشغيل المحدثة، من دون DeepSpeed أو تثبيت PyTorch قسراً. |
| `requirements_qwen_colab.txt` | نسخة صريحة من اعتماديات Colab نفسها. |
| `peft-mora/` | فرع PEFT المحلي الذي يضيف `use_mora=True` و`mora_type`. لم يُعدّل كود MoRA في هذه الحزمة. |
| `train_legacy_deepspeed.py` | نسخة احتياطية من نقطة الدخول القديمة الخاصة بـ LLaMA/DeepSpeed، ولا تُستخدم في المسار الجديد. |
| `colab_train_qwen25_3b_mora.ipynb` | دفتر Colab ينفذ الاستنساخ، التثبيت، تنزيل البيانات، الفحص، smoke test، التدريب، والاستئناف، مع حفظ محلي احتياطي عند تعذر Drive. |
| `.gitignore` | يمنع رفع الأسرار وملفات البيانات الكبيرة وcheckpoints إلى GitHub. |

ملف البيانات المضغوط هو النسخة التي جازت التدقيق السابق: **165,303 سجل**، وبحد أقصى مُدقّق يبلغ **1024 token** للسجل. السجلات تحتوي على الحقل `messages` بأدوار `system` و`user` و`assistant`، إلى جانب حقول المصدر والتصنيف والمرجع التي لا يحتاجها المدرب بعد مرحلة التحقق.

## التشغيل من الفرع العام وGoogle Colab

الفرع المخصص للمشروع هو `moranew` داخل المستودع العام `EngKHALIDx/qwen25-3b-yemeni-mora`. افتح ملف `colab_train_qwen25_3b_mora.ipynb` في Colab، ثم اختر GPU وشغّل الخلايا بالترتيب. لا تحتاج النسخة العامة الحالية إلى `GITHUB_TOKEN`. إذا أُعيدت خصوصية المستودع مستقبلاً، يمكن إضافة Secret باسم `GITHUB_TOKEN` يملك صلاحية القراءة، أو رفع مجلد المشروع يدوياً إلى `/content/qwen25-3b-yemeni-mora`. لا تضع أي مفتاح داخل الدفتر ولا ترفعه إلى GitHub.

الدفتر ينزّل الملف الخام من معرّف Drive المضمن في الخلية، ثم ينشئ نسخة موحّدة باسم `qwen25_3b_semantic_colab_train.jsonl` تحتوي على `messages` و`metadata_json` فقط، ويتحقق من SHA-256 وعدد السجلات قبل التدريب. لا تُرفع نسخة البيانات الكبيرة إلى GitHub؛ تُجلب وقت التشغيل من Drive. يحاول الدفتر حفظ النتائج في Google Drive، لكن عند فشل `drive.mount` يستخدم `/content/qwen25_3b_mora_adapter` محلياً ولا يوقف التدريب؛ بعده يمكن ضغط الـadapter وتنزيله من خلية التصدير الأخيرة.

## التشغيل السريع في Google Colab

ابدأ بتفعيل GPU من خلال `Runtime > Change runtime type > T4 GPU` أو V100، ثم ارفع ملف ZIP واستخرجه. إذا كان المشروع في `/content/mora_qwen_colab`، نفّذ الخلايا التالية:

```python
%cd /content/mora_qwen_colab
!nvidia-smi
!pip install -q -r requirements.txt
```

بعد اكتمال التثبيت شغّل تدريباً تجريبياً قصيراً للتأكد من أن tokenizer وPEFT والذاكرة تعمل قبل البدء الكامل:

```python
!python train.py \
  --config configs/qwen25_3b_mora.json \
  --max_train_samples 128 \
  --max_steps 5 \
  --output_dir ./smoke_test_adapter
```

إذا اكتمل الاختبار دون أخطاء، شغّل كامل مجموعة البيانات بالإعدادات المعتمدة. يفضل تشغيله من خلية الدفتر لأن مسار النتائج فيها يختار Drive إن توفر أو `/content` تلقائياً:

```python
!python train.py \\
  --config configs/qwen25_3b_mora.json \\
  --data_path /content/qwen25_data/qwen25_3b_semantic_colab_train.jsonl \\
  --output_dir /content/qwen25_3b_mora_adapter
```

يمكن تمرير أي قيمة من سطر الأوامر لتجاوز قيمة ملف JSON. مثال ذلك استخدام نسبة تحقق صغيرة مع الإبقاء على بقية الإعدادات:

```python
!python train.py \
  --config configs/qwen25_3b_mora.json \
  --validation_size 0.01 \
  --output_dir ./qwen25_3b_mora_adapter_with_eval
```

## لماذا لا تُقصّ السجلات؟

يرفض السكربت تلقائياً أي سجل يتجاوز `--max_seq_length` بدلاً من قصّه بصمت. هذا قرار مقصود لأن مجموعة البيانات الحالية أُنشئت ودُققت بحيث لا تتجاوز 1024 token، ولأن القص قد يحذف مادة أو قيداً أو مرجعاً قانونياً. عند استخدام ملف بيانات آخر يجب تقسيمه دلالياً قبل التدريب، أو رفع `--max_seq_length` إذا كانت الذاكرة تسمح، لا تمرير سجل مبتور دون تدقيق.

يحوّل السكربت كل `messages` إلى token IDs باستخدام قالب Qwen الرسمي، ثم يجعل labels الخاصة برسائل `system` و`user` تساوي `-100`. في مسار Colab يجب تمرير ملف `qwen25_3b_semantic_colab_train.jsonl` الموحد؛ أما الملف الخام ذي الأعمدة المتغيرة فلا يُمرر مباشرة إلى `datasets.load_dataset`. وبذلك يتعلم النموذج إنتاج إجابات المساعد مع بقاء السياق القانوني كاملاً أمامه. إذا تعذر تحديد نطاق المساعد بسبب قالب tokenizer غير متوافق، يتوقف السكربت بدلاً من إجراء تدريب بقناع loss خاطئ.

## الإعدادات المعتمدة للـ T4/V100

| الإعداد | القيمة الافتراضية | السبب العملي |
|---|---:|---|
| النموذج | `Qwen/Qwen2.5-3B-Instruct` | النموذج المطلوب في المشروع. |
| `use_4bit` | `true` | تحميل backbone بصيغة NF4 لتقليل ذاكرة T4. |
| `r` | `64` | رتبة MoRA عالية نسبياً مع كلفة ما زالت مناسبة للـ adapter. |
| `mora_type` | `6` | نوع MoRA المعتمد في إعداد المشروع. |
| الوحدات المستهدفة | الإسقاطات attention وMLP السبع | تغطية `q/k/v/o` و`gate/up/down` في بنية Qwen. |
| طول السجل | `1024` | مطابق لتدقيق ملف البيانات الحالي. |
| batch لكل GPU | `1` | يقلل الذروة في ذاكرة T4. |
| gradient accumulation | `16` | يرفع الحجم الفعال دون رفع batch الفعلي. |
| gradient checkpointing | مفعّل | يخفض الذاكرة مقابل وقت تدريب أطول. |
| attention | `sdpa` | لا يتطلب تثبيت `flash-attn` منفصلاً. |
| optimizer | `paged_adamw_8bit` | مناسب لتدريب adapter مع QLoRA. |
| learning rate | `2e-4` | نقطة بداية عملية لتدريب PEFT، ويمكن تغييرها بالتجربة. |

الإعداد الافتراضي يستخدم FP16 لأن T4 وV100 لا ينبغي افتراض دعمهما لـ BF16. إذا كانت البطاقة تدعم BF16 فعلياً، يمكن تمرير `--bf16`، وسيتحقق السكربت من الدعم قبل تفعيله. لا تمرر `--bf16` لمجرد أن نسخة PyTorch تعرض الخاصية؛ يجب أن تكون البطاقة نفسها داعمة.

## تثبيت MoRA والتحقق من عدم استخدام PEFT العادي

يجب أن يتم التثبيت من داخل مجلد المشروع حتى يستورد Python النسخة المحلية:

```bash
pip install -e ./peft-mora
```

عند `use_mora=true` يفحص السكربت وجود الحقل `use_mora` داخل `LoraConfig`. إذا كان المستورد يشير إلى PEFT العادي، يتوقف برسالة واضحة بدلاً من تدريب LoRA عادي وتسميته خطأً MoRA. تظل حزمة `peft-mora/` كما جاءت من المستودع الأصلي [2].

## استئناف التدريب والحفظ

بعد التدريب يحفظ `Trainer` adapter وtokenizer وملفات الحالة داخل `output_dir`. إذا كان Drive مربوطاً فاختر مساراً داخل `MyDrive`، وإلا استخدم `/content/qwen25_3b_mora_adapter` واضغط المجلد من خلية التصدير الأخيرة قبل انتهاء جلسة Colab. لاستئناف آخر checkpoint:

```python
!python train.py \
  --config configs/qwen25_3b_mora.json \
  --resume_from_checkpoint ./qwen25_3b_mora_adapter/checkpoint-500 \
  --output_dir ./qwen25_3b_mora_adapter
```

يُنشئ السكربت أيضاً `training_metadata.json`، ويتضمن النموذج الأساسي، مسار البيانات، طول السجل، نوع MoRA، الرتبة، الوحدات المستهدفة، وعدد السجلات المدربة. لا تحذف مجلدات `checkpoint-*` قبل التأكد من أن آخر حفظ اكتمل.

## استخدام الـ adapter بعد التدريب

للاختبار، حمّل النموذج الأساسي بكمية 4-bit ثم ركّب adapter المحفوظ:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = "./qwen25_3b_mora_adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

messages = [
    {
        "role": "system",
        "content": "أنت مساعد قانوني يمني مصدر-مقيد. لا تخترع مادة أو واقعة.",
    },
    {
        "role": "user",
        "content": "اشرح القاعدة القانونية في النص المرفق مع ذكر المرجع.",
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

ينبغي أثناء الاختبار تمرير نص قانوني أو سؤال مرتبط بالمجموعة المرجعية، ومراجعة أن الإجابة تميز بين النص المنقول والتحليل والوقائع التي تحتاج إثباتاً. لا يُفترض أن يحوّل التدريب المصدر المقيد إلى ترخيص لاختلاق قانون خارج `book.md`.

## معالجة مشاكل الذاكرة

الإعداد الحالي يبدأ بأقل batch فعلي ممكن وبـQLoRA وgradient checkpointing. إذا ظهر نفاد ذاكرة، تحقق أولاً من عدم وجود نموذج أو Tensor قديم في جلسة Colab، ثم أعد تشغيل runtime. بعد ذلك يمكن خفض `r` إلى 32 أو الاقتصار على وحدات attention، مع إدراك أن ذلك يقلل سعة التكيف. لا يُنصح بخفض `max_seq_length` لملف البيانات الحالي إلا بعد إعادة تدقيق دلالي، لأن خفضه قد يغيّر مضمون السجل.

إذا كان الخطأ مرتبطاً بـ`flash_attention_2`، اترك `attn_implementation` على `sdpa`. وإذا كان الخطأ متعلقاً بـBF16، احذف `--bf16` واترك الإعداد الافتراضي FP16. وإذا ظهر أن `use_mora` غير معروف، نفّذ تثبيت `-e ./peft-mora` من مجلد المشروع وأعد تشغيل نواة Colab حتى تزول نسخة PEFT القديمة من الذاكرة.

## المراجع

[1]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct "Qwen2.5-3B-Instruct model card"

[2]: https://github.com/kongds/MoRA "kongds/MoRA repository"

[3]: https://huggingface.co/docs/transformers/main/chat_templating "Hugging Face chat templates"

[4]: https://huggingface.co/docs/peft/developer_guides/quantization "PEFT quantization and QLoRA guidance"
