####################
# VANILLA FINETUNE #
####################
LA_DATA="french"
LA="fr"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="portuguese"
LA="pt"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="arabic"
LA="ar"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="italian"
LA="it"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="spanish"
LA="es"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"

LA_DATA="german"
LA="de"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "xlm-roberta-base" -o "ckpts/xlm-roberta-base-${LA}" --repo-id "cardiffnlp/xlm-roberta-base-tweet-sentiment-${LA}"


##########################
# TRIM FINE-TUNED MODELS #
##########################
trim_1 () {
  MODEL="${4}-tweet-sentiment-${1}-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "cardiffnlp/${4}-tweet-sentiment-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v "${3}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval.json"
  mv "${MODEL}" "best_model"
  mkdir "${MODEL}"
  mv "best_model" "${MODEL}"
  python experiments/finetune_multilabel.py --skip-train -n "${2}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
  mv "${MODEL}/eval.json" "${MODEL}/best_model"
  cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
  rm -rf "${MODEL}"
}

trim_2 () {
  MODEL="${3}-tweet-sentiment-${1}-trimmed-${1}"
  vocabtrimmer-trimming -m "cardiffnlp/${3}-tweet-sentiment-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval.json"
  mv "${MODEL}" "best_model"
  mkdir "${MODEL}"
  mv "best_model" "${MODEL}"
  python experiments/finetune_multilabel.py --skip-train -n "${2}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
  mv "${MODEL}/eval.json" "${MODEL}/best_model"
  cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
  rm -rf "${MODEL}"
}

LA_DATA="french"
LA="fr"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done

LA_DATA="portuguese"
LA="pt"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done

LA_DATA="arabic"
LA="ar"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done

LA_DATA="italian"
LA="it"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done


LA_DATA="spanish"
LA="es"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done


LA_DATA="german"
LA="de"
trim_2 ${LA} ${LA_DATA} "xlm-roberta-base"
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET} "xlm-roberta-base"
done

##########################
# FINETUNE TRIMMED XLM-R #
##########################
[HAWK]
LA_DATA="french"
LA="fr"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000 60000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

[HAWK]
LA_DATA="portuguese"
LA="pt"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000 60000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="italian"
LA="it"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000 60000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="spanish"
LA="es"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000 60000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="german"
LA="de"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000 60000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done

LA_DATA="arabic"
LA="ar"
vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}"
python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-tweet-sentiment-${LA}"
for TARGET in 5000 10000 15000 30000
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" -p "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}" -o "ckpts/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/xlm-roberta-base-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
done
