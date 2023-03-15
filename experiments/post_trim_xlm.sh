#####################
# FOUNDATION MODELS #
#####################
# xlm-r
LA="fr"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000 60000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done

LA="pt"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000 60000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done

LA="ar"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done

LA="it"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000 60000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done

LA="es"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000 60000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done

LA="de"
MODEL="xlm-roberta-base-trimmed-${LA}"
#vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
for TARGET in 5000 10000 15000 30000 60000
do
  MODEL="xlm-roberta-base-trimmed-${LA}-${TARGET}"
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
done


#####################
# FINE-TUNED MODELS #
#####################
trim_1 () {
  MODEL="xlm-roberta-base-tweet-sentiment-${1}-trimmed-${1}-${3}"
  vocabtrimmer-trimming -m "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v "${3}"
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
  MODEL="xlm-roberta-base-tweet-sentiment-${1}-trimmed-${1}"
  vocabtrimmer-trimming -m "vocabtrimmer/xlm-roberta-base-tweet-sentiment-${1}" -l "${1}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
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

LA_DATA="french"
LA="fr"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done

LA_DATA="portuguese"
LA="pt"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done

LA_DATA="arabic"
LA="ar"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done

LA_DATA="italian"
LA="it"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done

LA_DATA="spanish"
LA="es"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000 60000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done

LA_DATA="german"
LA="de"
#trim_2 ${LA} ${LA_DATA}
for TARGET in 5000 10000 15000 30000 60000 90000
do
  trim_1 ${LA} ${LA_DATA} ${TARGET}
done