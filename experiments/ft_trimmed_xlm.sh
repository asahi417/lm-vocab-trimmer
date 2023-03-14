################
# FINETUNE XLM #
################
LA_DATA="french"
LA="fr"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="portuguese"
LA="pt"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="arabic"
LA="ar"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="italian"
LA="it"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="spanish"
LA="es"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="german"
LA="de"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${LA}"

########################
# FINETUNE TRIMMED XLM #
########################
LA_DATA="french"
LA="fr"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000 60000 75000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="portuguese"
LA="pt"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000 60000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="arabic"
LA="ar"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="italian"
LA="it"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000 60000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="spanish"
LA="es"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000 60000 75000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="german"
LA="de"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 15000 30000 45000 60000 75000 90000
do
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA_DATA}-${TARGET}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

