# Fine-tuning trimmed multilingual LM (mt5 small/base) on QG. Instead of running grid search, we use the optimal
# configuration used in the original fine-tuning

########
# BASE #
########
SIZE="base"
LA='ja'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 16 -g 4 --lr 0.0001 --label-smoothing 0.0 --random-seed 1 -e 30
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_30" -e "lmqg_output/trim/${MODEL}/epoch_30/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_30/" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='ko'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 16 -g 4 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 11
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_11" -e "lmqg_output/trim/${MODEL}/epoch_11/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_11" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='ru'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 8 -g 8 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 16
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_16" -e "lmqg_output/trim/${MODEL}/epoch_16/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_16" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='fr'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 8 -g 8 --lr 0.0001 --label-smoothing 0.15 --random-seed 1 -e 24
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_24" -e "lmqg_output/trim/${MODEL}/epoch_24/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_24" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='de'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 8 -g 8 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 17
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_17" -e "lmqg_output/trim/${MODEL}/epoch_17/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_17" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='es'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 8 -g 8 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 10
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_10" -e "lmqg_output/trim/${MODEL}/epoch_10/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_10" -a "${MODEL}" -o "lmqg"

SIZE="base"
LA='it'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 8 -g 8 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 11
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_11" -e "lmqg_output/trim/${MODEL}/epoch_11/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_11" -a "${MODEL}" -o "lmqg"


#########
# SMALL #
#########
SIZE="small"
LA='ja'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 64 -g 1 --lr 0.0005 --label-smoothing 0.0 --random-seed 1 -e 21
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_21" -e "lmqg_output/trim/${MODEL}/epoch_21/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_21/" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='ko'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 64 -g 1 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 7
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_7" -e "lmqg_output/trim/${MODEL}/epoch_7/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_7" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='ru'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 5
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_5" -e "lmqg_output/trim/${MODEL}/epoch_5/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_5" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='fr'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 14
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_14" -e "lmqg_output/trim/${MODEL}/epoch_14/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_14" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='de'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 11
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_11" -e "lmqg_output/trim/${MODEL}/epoch_11/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_11" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='es'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 16
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_16" -e "lmqg_output/trim/${MODEL}/epoch_16/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_16" -a "${MODEL}" -o "lmqg"

SIZE="small"
LA='it'
MODEL="mt5-${SIZE}-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-${SIZE}-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.0005 --label-smoothing 0.0 --random-seed 1 -e 15
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_15" -e "lmqg_output/trim/${MODEL}/epoch_15/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_15" -a "${MODEL}" -o "lmqg"
