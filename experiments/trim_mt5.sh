#####################
# FOUNDATION MODELS #
#####################
# mt5
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}"
  done
done

for LA in 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    for TARGET in 15000 30000 45000 60000
    do
      vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -v "${TARGET}"
    done
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    for TARGET in 75000 90000 105000
    do
      vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-${TARGET}" -v "${TARGET}"
    done
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es'
do
  for SIZE in "small" "base"
  do
    vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/mt5-${SIZE}-trimmed-${LA}-120000" -p "ckpts/mt5-${SIZE}-trimmed-${LA}-120000" -v 120000
  done
done



#################
# TASK SPECIFIC #
#################
# mt5: NEED TO INSTALL `lmqg` https://github.com/asahi417/lm-question-generation
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
    rm -rf "${MODEL}"
    rm -rf "ckpts/${MODEL}"
  done
done

for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    for TARGET in 15000 30000 45000 60000
    do
      MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${TARGET}"
      vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
      git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
      rm -rf "${MODEL}/eval"
      lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
      cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
      rm -rf "${MODEL}"
      rm -rf "ckpts/${MODEL}"
    done
  done
done


for LA in 'ja' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    for TARGET in 75000 90000 105000
    do
      MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${TARGET}"
      vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
      git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
      rm -rf "${MODEL}/eval"
      lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
      cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
      rm -rf "${MODEL}"
      rm -rf "ckpts/${MODEL}"
    done
  done
done

for LA in 'ja' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-120000"
    vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v 120000
    git clone "https://huggingface.co/vocabtrimmer/${MODEL}"
    rm -rf "${MODEL}/eval"
    lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
    rm -rf "${MODEL}"
    rm -rf "ckpts/${MODEL}"
  done
done



