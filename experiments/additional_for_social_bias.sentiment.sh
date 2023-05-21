LM="xlm-roberta-base"
LM_ALIAS="xlm-roberta-base"
TARGET=50000
LA_DATA="english"
LA="en"
vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" --target-vocab-size "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
python experiments/finetune_sentiment.py -n "${LA_DATA}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -o "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}-tweet-sentiment-${LA}"
rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"


trim_1 () {
  MODEL="${4}-tweet-sentiment-${1}-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "cardiffnlp/${4}-tweet-sentiment-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v "${3}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval.json"
  mv "${MODEL}" "best_model"
  mkdir "${MODEL}"
  mv "best_model" "${MODEL}"
  python experiments/finetune_sentiment.py --skip-train -n "${2}" -m "vocabtrimmer/${MODEL}" -o "${MODEL}"
  mv "${MODEL}/eval.json" "${MODEL}/best_model"
  cd "${MODEL}/best_model" && git add . && git commit -m "update" && git push && cd ../../
  rm -rf "${MODEL}"
}

trim_1 ${LA} ${LA_DATA} ${TARGET} "${LM_ALIAS}"

