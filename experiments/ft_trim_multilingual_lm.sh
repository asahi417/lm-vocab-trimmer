
LA='ja'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 64 -g 1 --lr 0.0005 --label-smoothing 0.0 --random-seed 1 -e 21
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_21" -e "lmqg_output/trim/${MODEL}/epoch_21/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_21/" -a "${MODEL}" -o "lmqg"


LA='ko'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 64 -g 1 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 7
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_7" -e "lmqg_output/trim/${MODEL}/epoch_7/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_7" -a "${MODEL}" -o "lmqg"


LA='ru'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 5
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_5" -e "lmqg_output/trim/${MODEL}/epoch_5/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_5" -a "${MODEL}" -o "lmqg"


LA='fr'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 14
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_14" -e "lmqg_output/trim/${MODEL}/epoch_14/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_14" -a "${MODEL}" -o "lmqg"


LA='de'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.001 --label-smoothing 0.15 --random-seed 1 -e 11
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_11" -e "lmqg_output/trim/${MODEL}/epoch_11/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_11" -a "${MODEL}" -o "lmqg"



LA='es'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.0005 --label-smoothing 0.15 --random-seed 1 -e 16
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_16" -e "lmqg_output/trim/${MODEL}/epoch_16/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_16" -a "${MODEL}" -o "lmqg"


LA='it'
MODEL="mt5-small-trimmed-${LA}-${LA}quad-qg"
lmqg-train -c "lmqg_output/trim/${MODEL}" -d "lmqg/qg_${LA}quad" -m "asahi417/mt5-small-trimmed-${LA}" --low-cpu-mem-usage \
-b 32 -g 2 --lr 0.0005 --label-smoothing 0.0 --random-seed 1 -e 15
lmqg-eval -m "lmqg_output/trim/${MODEL}/epoch_15" -e "lmqg_output/trim/${MODEL}/epoch_15/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer" --prediction-aggregation "first" --prediction-level "sentence"
lmqg-push-to-hf -m "lmqg_output/trim/${MODEL}/epoch_15" -a "${MODEL}" -o "lmqg"
