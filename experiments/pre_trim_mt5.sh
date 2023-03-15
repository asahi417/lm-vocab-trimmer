########################
# QG FINE-TUNED MODELS #
########################
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "vocabtrimmer/mt5-small-trimmed-${LA}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
  lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 5000 10000 15000 30000 60000
  do
    LA=es
    TARGET=30000
    MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qg"
    lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
    lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
    lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"with
  done
done

[STONE]
for LA in 'ja' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-trimmed-${LA}-90000-${LA}quad-qg"
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "vocabtrimmer/mt5-small-trimmed-${LA}-90000" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
  lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

[UKRI]
for LA in 'ja' 'ru' 'fr' 'es'
do
  MODEL="mt5-small-trimmed-${LA}-120000-${LA}quad-qg"
  lmqg-train-search -c "lmqg_output/trimmed_qg/${MODEL}" -d "lmqg/qg_${LA}quad" -m "vocabtrimmer/mt5-small-trimmed-${LA}-120000" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
  lmqg-eval -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -e "lmqg_output/trimmed_qg/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qg/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

########################
# QA FINE-TUNED MODELS #
########################
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-trimmed-${LA}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 5000 10000 15000 30000 60000
  do
#    LA=es
#    TARGET=60000
    MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qg"
    lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
    lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
    lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
    lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
  done
done

for LA in 'ja' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-trimmed-${LA}-90000-${LA}quad-qg"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

for LA in 'ja' 'ru' 'fr' 'es'
do
  MODEL="mt5-small-trimmed-${LA}-120000-${LA}quad-qg"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done
