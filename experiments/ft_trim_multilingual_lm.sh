# Fine-tuning trimmed multilingual LM (mt5 small/base) on QG
SIZE="small"
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
  lmqg-train-search -c "lmqg_output/trimmed/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" -b 32 -g 2 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
  lmqg-eval -m "lmqg_output/trimmed/${MODEL}/best_model" -e "lmqg_output/trimmed/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed/${MODEL}/best_model" -a "${MODEL}" -o "lmqg"
done

SIZE="base"
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
  lmqg-train-search -c "lmqg_output/trimmed/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" -b 16 -g 4 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
#  lmqg-train-search -c "lmqg_output/trimmed/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" -b 32 -g 2 --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1
  lmqg-eval -m "lmqg_output/trimmed/${MODEL}/best_model" -e "lmqg_output/trimmed/${MODEL}/best_model/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
  lmqg-push-to-hf -m "lmqg_output/trimmed/${MODEL}/best_model" -a "${MODEL}" -o "lmqg"
done