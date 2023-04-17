sentiment () {
  LM=${1}
  LM_ALIAS=${2}

  ####################
  # VANILLA FINETUNE #
  ####################
  for LA in "en" "fr" "de" "es" "ar"
  do
    python experiments/finetune_xnli.py -n "${LA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-xnli-${LA}"
  done

  ##########################
  # TRIM FINE-TUNED MODELS #
  ##########################
#  trim_1 () {
#    MODEL="${3}-xnli-${1}-trimmed-${1}-${2}"
#    vocabtrimmer-trimming -m "vocabtrimmer/${3}-xnli-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v "${2}"
#    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
#    rm -rf "${MODEL}/eval.json"
#    mv "${MODEL}" "best_model"
#    mkdir "${MODEL}"
#    mv "best_model" "${MODEL}"
#    python experiments/finetune_xnli.py --skip-train -n "${1}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
#    mv "${MODEL}/eval.json" "${MODEL}/best_model"
#    cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
#    rm -rf "${MODEL}"
#  }

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

#  for LA in "en" "fr" "de" "es" "ar"
#  do
#    for TARGET in 5000 10000 15000 30000 60000
#    do
#      trim_1 ${LA} ${TARGET} "${LM_ALIAS}"
#    done
#  done
#
#  LA="ar"
#  for TARGET in 5000 10000 15000 30000
#  do
#    trim_1 ${LA} ${LA} ${TARGET} "${LM_ALIAS}"
#  done

  ##########################
  # FINETUNE TRIMMED XLM-R #
  ##########################
  # ar [hawk]
  # en [hawk]
  # de [hawk]
  # es [hawk]
  # ar [hawk]
  for LA in "en" "fr" "de" "es" "ar"
  do
#    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
    python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}"
  done

#  for LA in "en" "fr" "de" "es" "ar"
#  do
#    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
#    python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"
#    rm -rf "${LM_ALIAS}-trimmed-${LA}"
#    for TARGET in 5000 10000 15000 30000 60000
#    do
#      vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
#      python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
#      rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
#      rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
#    done
#  done

#  [hawk]
#  LA="ar"
##  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
#  python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-xnli-${LA}"
#  rm -rf "${LM_ALIAS}-trimmed-${LA}"
#  for TARGET in 5000 10000 15000 30000 60000
#  do
#    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
#    python experiments/finetune_xnli.py -n "${LA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
#    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-xnli-${LA}"
#    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
#  done
}

sentiment "xlm-roberta-base" "xlm-roberta-base"
sentiment "facebook/xlm-v-base" "xlm-v-base"
LM="xlm-roberta-base"
LM_ALIAS="xlm-roberta-base"

LM="facebook/xlm-v-base"
LM_ALIAS="xlm-v-base"


python experiments/finetune_xnli.py -n "en" -m "roberta-base" -o "ckpts/roberta-base-en" --repo-id "vocabtrimmer/roberta-base-xnli-en"
python experiments/finetune_xnli.py -n "fr" -m "camembert-base" -o "ckpts/camembert-base-fr" --repo-id "vocabtrimmer/camembert-base-xnli-fr"
