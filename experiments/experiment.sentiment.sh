sentiment () {
  LM=${1}
  LM_ALIAS=${2}

  ####################
  # VANILLA FINETUNE #
  ####################
  LA_DATA="english"
  LA="en"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="french"
  LA="fr"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="portuguese"
  LA="pt"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="arabic"
  LA="ar"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="italian"
  LA="it"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="spanish"
  LA="es"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

  LA_DATA="german"
  LA="de"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${LM}" -o "ckpts/${LM_ALIAS}-${LA}" --repo-id "cardiffnlp/${LM_ALIAS}-tweet-sentiment-${LA}"

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

  # RUNNNIG ON HAWK #
  LA_DATA="english"
  LA="en"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="french"
  LA="fr"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="portuguese"
  LA="pt"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="arabic"
  LA="ar"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="italian"
  LA="it"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="spanish"
  LA="es"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done

  LA_DATA="german"
  LA="de"
  trim_2 ${LA} ${LA_DATA} "${LM_ALIAS}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"
  done
  # RUNNNIG ON HAWK #

  ##########################
  # FINETUNE TRIMMED XLM-R #
  ##########################
  LA_DATA="english"
  LA="en"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000 
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="french"
  LA="fr"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="portuguese"
  LA="pt"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="italian"
  LA="it"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="spanish"
  LA="es"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="german"
  LA="de"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done

  LA_DATA="arabic"
  LA="ar"
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-tweet-sentiment-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"
  for TARGET in 5000 10000 15000 30000 60000
  do
    vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
    python experiments/finetune_multilabel.py -n "${LA_DATA}" -m "${PWD}/ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
    rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  done
}

sentiment "xlm-roberta-base" "xlm-roberta-base"
sentiment "facebook/xlm-v-base" "xlm-v-base"
#  LM="facebook/xlm-v-base" "xlm-v-base"
#  LM_ALIAS="xlm-v-base"
#
#  LM="xlm-roberta-base"
#  LM_ALIAS="xlm-roberta-base"

