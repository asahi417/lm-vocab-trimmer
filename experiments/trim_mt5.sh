#####################
# FOUNDATION MODELS #
#####################
# mt5
SIZE="small"
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}"
done

for LA in 'fr' 'de' 'es' 'it'
do
  for TARGET in 15000 30000 45000 60000
  do
    vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -v "${TARGET}"
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es' 'it'
do
  for TARGET in 75000 90000 105000
  do
    vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -v "${TARGET}"
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es'
do
  vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-120000" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-120000" -v 120000
done


########################
# QG FINE-TUNED MODELS #
########################
# mt5: NEED TO INSTALL `lmqg` https://github.com/asahi417/lm-question-generation
SIZE="small"
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${LA}"
  vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
done

for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for TARGET in 15000 30000 45000 60000
  do
    MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es' 'it'
do
  for TARGET in 75000 90000 105000
  do
    MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es'
do
  MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${LA}-120000"
  vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v 120000
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
done


########################
# QA FINE-TUNED MODELS #
########################
# mt5: NEED TO INSTALL `lmqg` https://github.com/asahi417/lm-question-generation
SIZE="small"
for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  MODEL="mt5-${SIZE}-${LA}quad-qa-trimmed-${LA}"
  vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qa" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
done


for LA in 'ja' 'ko' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 15000 30000 45000 60000
  do
    MODEL="mt5-${SIZE}-${LA}quad-qa-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qa" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
    lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
    rm -rf "${MODEL}"
  done
done

for LA in 'ja' 'ru' 'fr' 'es' 'it'
do
  for TARGET in 75000 90000 105000
  do
    MODEL="mt5-${SIZE}-${LA}quad-qa-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qa" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
    lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
    rm -rf "${MODEL}"
  done
done

for LA in 'ja' 'ru' 'fr' 'es'
do
  MODEL="mt5-${SIZE}-${LA}quad-qa-trimmed-${LA}-120000"
  vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qa" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v 120000
  git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
  rm -rf "${MODEL}/eval"
  lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i 'paragraph_question' -o 'answer'
  lmqg-eval-qa -m "${MODEL}" -e "${MODEL}/eval" -d "lmqg/qg_${LA}quad" --language "${LA}"
  cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
  rm -rf "${MODEL}"
done



