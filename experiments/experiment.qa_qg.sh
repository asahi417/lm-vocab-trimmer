#############################
# TRIM FINE-TUNED mT5/mBART #
#############################
trim_1 () {
  if [[ "${1}" == "en" ]]; then
    DATASET="squad"
  else
    DATASET="${1}quad"
  fi
  # QG
  MODEL="${2}-${DATASET}-qg-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "lmqg/${2}-${DATASET}-qg" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${3}
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${DATASET}" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
  # QA
  MODEL="${2}-${DATASET}-qa-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "lmqg/${2}-${DATASET}-qa" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${3}
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${DATASET}" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${DATASET}" --language "${1}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
}

trim_2 () {
  if [[ "${1}" == "en" ]]; then
    DATASET="squad"
  else
    DATASET="${1}quad"
  fi
#  # QG
#  MODEL="${2}-${DATASET}-qg-trimmed-${1}"
#  vocabtrimmer-trimming -m "lmqg/${2}-${DATASET}-qg" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
#  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
#  rm -rf "${MODEL}/eval"
#  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${DATASET}" -i "paragraph_answer"
#  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
#  rm -rf "${MODEL}"
  # QA
  MODEL="${2}-${DATASET}-qa-trimmed-${1}"
  vocabtrimmer-trimming -m "lmqg/${2}-${DATASET}-qa" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${DATASET}" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${DATASET}" --language "${1}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
}

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it' 'en'
do
  trim_2 ${LA} 'mt5-small'
  trim_2 ${LA} 'mbart-large-cc25'
done

trim_2 en 'mbart-large-cc25'

# MT5
for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  trim_1 "en" "mt5-small" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  trim_1 "es" "mt5-small" ${TARGET}
done


for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  trim_1 "fr" "mt5-small" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  trim_1 "ja" "mt5-small" ${TARGET}
done


for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  trim_1 "ru" "mt5-small" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000 90000
do
  trim_1 "it" "mt5-small" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "ko" "mt5-small" ${TARGET}
done

# MBART
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "en" "mbart-large-cc25" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "es" "mbart-large-cc25" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "fr" "mbart-large-cc25" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000 90000
do
  trim_1 "ru" "mbart-large-cc25" ${TARGET}
done
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "ja" "mbart-large-cc25" ${TARGET}
done
for TARGET in 5000 10000 15000 30000
do
  trim_1 "ko" "mbart-large-cc25" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 "it" "mbart-large-cc25" ${TARGET}
done


########################
# FINETUNE TRIMMED mT5 #
########################
LM="facebook/mbart-large-cc25"
LM_ALIAS="mbart-large-cc25"
#
#LM="google/mt5-small"
#LM_ALIAS="mt5-small"
ft_1 () {
  LA=${1}
  TARGET=${2}

  # trim
  vocabtrimmer-trimming -m ${LM} -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -v "${TARGET}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}-${TARGET}"

  if [[ "${1}" == "en" ]]; then
    DATASET="squad"
  else
    DATASET="${1}quad"
  fi

  # finetune qg
  MODEL="${LM_ALIAS}-trimmed-${LA}-${TARGET}-${DATASET}-qg"
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${DATASET}" -m "ckpts/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${DATASET}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
  rm -rf "${MODEL}"

  # finetune qa
  MODEL="${LM_ALIAS}-trimmed-${LA}-${TARGET}-${DATASET}-qa"
  if [[ "${LA}" == "ja" ]]; then
    lmqg-train-search -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${DATASET}" --lr 8e-04 6e-04 4e-04 2e-04 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "${LA}" --n-max-config 1 -b 16 -g 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  else
    lmqg-train-search -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${DATASET}" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  fi
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${DATASET}" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${DATASET}" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
  rm -rf "${MODEL}"
}

ft_2 () {
  LA=${1}

  # trim
  vocabtrimmer-trimming -m "${LM}" -l "${LA}" -p "ckpts/${LM_ALIAS}-trimmed-${LA}" --repo-id "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}"
  rm -rf "${LM_ALIAS}-trimmed-${LA}"

  if [[ "${1}" == "en" ]]; then
    DATASET="squad"
  else
    DATASET="${1}quad"
  fi

  # finetune qg
  MODEL="${LM_ALIAS}-trimmed-${LA}-${DATASET}-qg"
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${DATASET}" -m "ckpts/${LM_ALIAS}-trimmed-${LA}" -b 32 -g 2 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${DATASET}" -m "ckpts/${LM_ALIAS}-trimmed-${LA}" -b 8 -g 8 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${DATASET}" -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -b 8 -g 8 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${DATASET}" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
  rm -rf "${MODEL}"
  # finetune qa
  MODEL="${LM_ALIAS}-trimmed-${LA}-${DATASET}-qa"
  lmqg-train-search -m "vocabtrimmer/${LM_ALIAS}-trimmed-${LA}" -d "lmqg/qg_${DATASET}" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${DATASET}" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${DATASET}" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
  rm -rf "${MODEL}"
}


for LA in 'ko' 'ja' 'ru' 'fr' 'it' 'es' 'en'
do
  ft_2 ${LA}
done


for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "fr" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "ja" ${TARGET}
done


for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "en" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "es" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "ru" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000 90000
do
  ft_1 "it" ${TARGET}
done

for TARGET in 5000 10000 15000 30000 60000
do
  ft_1 "ko" ${TARGET}
done


# monolingual

lmqg-train-search -m "t5-small" -d "lmqg/qg_squad" --lr 5e-05 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0.15 --language "en" --n-max-config 1 -b 32 -g 2 -c "lmqg_output/qa/t5-small-squad" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval -m "lmqg_output/qa/t5-small-squad/best_model" -e "lmqg_output/qa/t5-small-squad/best_model/eval" -d "lmqg/qg_squad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/qa/t5-small-squad/best_model" -e "lmqg_output/qa/t5-small-squad/best_model/eval" -d "lmqg/qg_squad" --language "en"
lmqg-push-to-hf -m "lmqg_output/qa/t5-small-squad/best_model" -a "t5-small-squad-qa" -o "lmqg"
