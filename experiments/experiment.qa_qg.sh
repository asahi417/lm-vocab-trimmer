#############################
# TRIM FINE-TUNED mT5/mBART #
#############################
trim_1 () {
  # QG
#  MODEL="${2}-${1}quad-qg-trimmed-${1}-${3}"
#  vocabtrimmer-trimming -m "lmqg/${2}-${1}quad-qg" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${3}
#  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
#  rm -rf "${MODEL}/eval"
#  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${1}quad" -i "paragraph_answer"
#  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
#  rm -rf "${MODEL}"
  # QA
  MODEL="${2}-${1}quad-qa-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "lmqg/${2}-${1}quad-qa" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${3}
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${1}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${1}quad" --language "${1}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
}

trim_2 () {
  # QG
#  MODEL="${2}-${1}quad-qg-trimmed-${1}"
#  vocabtrimmer-trimming -m "lmqg/${2}-${1}quad-qg" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
#  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
#  rm -rf "${MODEL}/eval"
#  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${1}quad" -i "paragraph_answer"
#  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
#  rm -rf "${MODEL}"
  # QA
  MODEL="${2}-${1}quad-qa-trimmed-${1}"
  vocabtrimmer-trimming -m "lmqg/${2}-${1}quad-qa" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${1}" -d "lmqg/qg_${1}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${1}quad" --language "${1}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
}
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  trim_2 ${LA} 'mt5-small'
  trim_2 ${LA} 'mbart-large-cc25'
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
ft_1 () {
    LA=${1}
    TARGET=${2}
#    # trim
#    vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" -p "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -v "${TARGET}"
#    rm -rf "mt5-small-trimmed-${LA}-${TARGET}"
#    # finetune qg
#    MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qg"
#    lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
##    lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -b 64 -g 1 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
#    lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
#    lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
#    rm -rf "${MODEL}"
    # finetune qa
    MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
    lmqg-train-search -m "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
#    lmqg-train-search -m "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 64 -g 1 2 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
    lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
    lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
    lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
    rm -rf "${MODEL}"
}

ft_2 () {
    LA=${1}
    # trim
    vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" -p "ckpts/mt5-small-trimmed-${LA}"
    rm -rf "mt5-small-trimmed-${LA}"
    # finetune qg
    MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
    lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "ckpts/mt5-small-trimmed-${LA}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 --low-cpu-mem-usage
    lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
    lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
    rm -rf "${MODEL}"
    # finetune qa
    MODEL="mt5-small-trimmed-${LA}-${LA}quad-qa"
    lmqg-train-search -m "ckpts/mt5-small-trimmed-${LA}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
    lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
    lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
    lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
    rm -rf "${MODEL}"
}


#torun
ft_1 "ru" 120000
#running
ft_1 "fr" 60000
ft_1 "ru" 30000
ft_1 "it" 30000
ft_1 "ko" 30000
ft_1 "it" 15000
ft_1 "ko" 60000
ft_1 "it" 60000
ft_1 "ru" 60000
ft_1 "es" 60000
ft_1 "es" 120000
ft_1 "fr" 90000
ft_1 "es" 15000
ft_1 "es" 30000
# ukri
ft_1 "ko" 15000

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  ft_2 ${LA}
done

for TARGET in 5000 10000 15000 30000 60000 90000 120000
do
  ft_1 "es" ${TARGET}
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
