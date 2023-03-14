# Fine-tuning trimmed multilingual LM (mt5 small) on QA
for LA in 'ja' 'ko' 'es' 'it' 'ru' 'fr'
do
  MODEL="mt5-small-trimmed-${LA}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

# TARGET=120000 105000 90000 75000 60000 45000 15000 30000
LA="es"
for TARGET in 15000 30000 45000 60000 75000 90000 105000 120000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

LA="it"
#TARGET=15000 30000 45000 6000075000 90000 105000
for TARGET in 15000 30000 45000 60000 75000 90000 105000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

LA="ko"
#TARGET=15000 30000 45000 60000
for TARGET in 15000 30000 45000 60000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

LA="ru"
#TARGET=15000 30000 45000 60000 75000 90000 105000 120000
for TARGET in 15000 30000 45000 60000 75000 90000 105000 120000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

LA="fr"
# TARGET=15000 30000 45000 60000 75000 90000 105000 120000
for TARGET in 15000 30000 45000 60000 75000 90000 105000 120000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

# On STONE
LA="ja"
#TARGET=105000
for TARGET in 15000 30000 45000 60000 75000 90000 105000 120000
do
  MODEL="mt5-small-trimmed-${LA}-${TARGET}-${LA}quad-qa"
  lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
  lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
done

# TODO
LA="it"
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qa"
lmqg-train-search -m "vocabtrimmer/mt5-small-trimmed-${LA}" -d "lmqg/qg_${LA}quad" --lr 1e-04 5e-04 1e-03 --epoch-partial 5 -e 15 --label-smoothing 0 0.15 --language "${LA}" --n-max-config 1 -b 32 -g 2 4 -c "lmqg_output/trimmed_qa/${MODEL}" -i 'paragraph_question' -o 'answer' --low-cpu-mem-usage
lmqg-eval -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
lmqg-eval-qa -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -e "lmqg_output/trimmed_qa/${MODEL}/best_model/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
lmqg-push-to-hf -m "lmqg_output/trimmed_qa/${MODEL}/best_model" -a "${MODEL}" -o "vocabtrimmer"
