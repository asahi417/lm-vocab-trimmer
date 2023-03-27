#####################
# FOUNDATION MODELS #
#####################
# mt5
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "vocabtrimmer/mt5-small-trimmed-${LA}" -p "ckpts/mt5-small-trimmed-${LA}"
done

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 5000 10000 15000 30000 60000
  do
    TARGET=15000
    LA="fr"
    vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "vocabtrimmer/mt5-small-trimmed-${LA}-${TARGET}" -p "ckpts/mt5-small-trimmed-${LA}-${TARGET}" -v "${TARGET}"
    rm -rf "mt5-small-trimmed-${LA}-${TARGET}"
  done
done

for LA in 'ja' 'ru' 'fr' 'es' 'it'
do
  vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "vocabtrimmer/mt5-small-trimmed-${LA}-90000" -p "ckpts/mt5-small-trimmed-${LA}-90000" -v 90000
  rm -rf "mt5-small-trimmed-${LA}-90000"
done

for LA in 'ja' 'ru' 'fr' 'es'
do
  vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "vocabtrimmer/mt5-small-trimmed-${LA}-120000" -p "ckpts/mt5-small-trimmed-${LA}-120000" -v 120000
  rm -rf "mt5-small-trimmed-${LA}-120000"
done

#mbart
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  vocabtrimmer-trimming -m "facebook/mbart-large-cc25" -l "${LA}" --repo-id "vocabtrimmer/mbart-large-cc25-trimmed-${LA}" -p "ckpts/mbart-large-cc25-trimmed-${LA}"
done


########################
# QG FINE-TUNED MODELS #
########################
# mt5: NEED TO INSTALL `lmqg` https://github.com/asahi417/lm-question-generation
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-${LA}quad-qg-trimmed-${LA}"
  vocabtrimmer-trimming -m "lmqg/mt5-small-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
done

for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 5000 10000 15000 30000 60000
  do
    MODEL="mt5-small-${LA}quad-qg-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "lmqg/mt5-small-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
    rm -rf "${MODEL}"
  done
done

for LA in 'ja' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-small-${LA}quad-qg-trimmed-${LA}-90000"
  vocabtrimmer-trimming -m "lmqg/mt5-small-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v 90000
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
done

for LA in 'ja' 'ru' 'fr' 'es'
do
  MODEL="mt5-small-${LA}quad-qg-trimmed-${LA}-120000"
  vocabtrimmer-trimming -m "lmqg/mt5-small-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v 120000
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
done



