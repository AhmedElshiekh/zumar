# عرض المعلمين المتاحين
cargo run -- list-teachers

# استخراج logits لمعلم واحد
cargo run -- extract /data --teacher llama-13b

# تقطير من معلم واحد
cargo run -- distill 50 /data --teachers llama-13b

# عرض حالة التقطير
cargo run -- resume

# استخراج معلم ثانٍ
cargo run -- extract /data --teacher jais-13b --force

# تقطير من معلمين (يستأنف تلقائياً)
cargo run -- distill 100 /data --teachers llama-13b,jais-13b

# إعادة تعيين checkpoint (لبدء جديد)
cargo run -- reset


# نموذج صغير 80M (افتراضي)
cargo run -- distill 100 /data

# نموذج 400M
cargo run -- distill 100 /data --size 0.4B

# نموذج 1.5B
cargo run -- distill 100 /data --size 1.5B

# نموذج 7B
cargo run -- distill 100 /data --size 7B

# نموذج 13B
cargo run -- distill 100 /data --size 13B

# نموذج 70B
cargo run -- distill 100 /data --size 70B

# أبعاد مخصصة (hidden, layers, experts)
cargo run -- distill 100 /data --dims 2048 32 12

# استخراج logits من معلم 70B
cargo run -- extract /data --teacher llama-70b

# تقطير تسلسلي من معلمين مختلفي الحجم
cargo run -- distill 100 /data --teachers llama-70b,jais-13b