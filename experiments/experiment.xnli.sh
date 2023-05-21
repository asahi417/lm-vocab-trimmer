nli () {
  LM=${1}
  LM_ALIAS=${2}

  ####################
  # VANILLA FINETUNE #
  ####################
  for LA in "en" "fr" "de" "es" "ar"
  do
    python experiments/finetune_xnli.py -n "${LA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-xnli-${LA}"
  done

  trim_1 () {
    MODEL="${2}-xnli-${1}-trimmed-${1}-${3}"
    vocabtrimmer-trimming -m "vocabtrimmer/${2}-xnli-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v "${3}"
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval.json"
    mv "${MODEL}" "best_model"
    mkdir "${MODEL}"
    mv "best_model" "${MODEL}"
    python experiments/finetune_xnli.py --skip-train -n "${1}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
    mv "${MODEL}/eval.json" "${MODEL}/best_model"
    cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
    rm -rf "${MODEL}"
  }
  trim_2 () {
    MODEL="${2}-xnli-${1}-trimmed-${1}"
    vocabtrimmer-trimming -m "vocabtrimmer/${2}-xnli-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval.json"
    mv "${MODEL}" "best_model"
    mkdir "${MODEL}"
    mv "best_model" "${MODEL}"
    python experiments/finetune_xnli.py --skip-train -n "${1}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
    mv "${MODEL}/eval.json" "${MODEL}/best_model"
    cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
    rm -rf "${MODEL}"
  }
  for LA in "en" "fr" "de" "es" "ar"
  do
    trim_2 ${LA} "${LM_ALIAS}"
  done

  for LA in "en" "fr" "de" "es" "ar"
#  for LA in "de" "es"
  do
    for TARGET in 5000 10000 15000 30000 60000
    do
      trim_1 ${LA} "${LM_ALIAS}" "${TARGET}"
    done
  done

  ##########################
  # FINETUNE TRIMMED XLM-R #
  ##########################
  for LA in "en" "fr" "de" "es" "ar"
  do
#    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
    python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}"
  done

  for LA in "en" "fr" "de" "es" "ar"
  do
    for TARGET in 5000 10000 15000 30000 60000
    do
      python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
      rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
    done
  done


}

nli "xlm-roberta-base" "xlm-roberta-base"
nli "facebook/xlm-v-base" "xlm-v-base"

LM="facebook/xlm-v-base"
LM_ALIAS="xlm-v-base"
LA="en"
vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
python experiments/finetune_xnli.py --lr 0.00001 --epoch 20 -n "${LA}" -m "ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}-2" # --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"
python experiments/finetune_xnli.py --skip-train --skip-eval -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}-2" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"


# mono-lingual
python experiments/finetune_xnli.py -n "en" -m "roberta-base" -o "ckpts/roberta-base-en" --repo-id "vocabtrimmer/roberta-base-xnli-en"