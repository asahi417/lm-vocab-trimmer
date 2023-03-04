#####################
# FOUNDATION MODELS #
#####################
# mt5

HF_ORG='vocabtrimmer'
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "${HF_ORG}/mt5-${SIZE}-trimmed-${LA}" -p "ckpts/mt5-${SIZE}-trimmed-${LA}"
  done
done


HF_ORG='vocabtrimmer'
LA="ko"
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
#    for TARGET in 15000 30000 45000 60000 75000 90000 120000 150000 180000 210000
    for TARGET in 90000 120000 150000 180000 210000
    do
      MODEL="mt5-${SIZE}-trimmed-${LA}-${TARGET}"
      vocabtrimmer-trimming -m "google/mt5-${SIZE}" -l "${LA}" --repo-id "${HF_ORG}/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    done
  done
done

# xlm-r
HF_ORG='vocabtrimmer'
for LA in 'ja' 'fr' 'de' 'es' 'it' 'ar' 'pt'
do
  for SIZE in "base" "large"
  do
    for TARGET in 15000 30000 45000 60000 75000
    do
      MODEL="xlm-roberta-${SIZE}-trimmed-${LA}-${TARGET}"
      vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "${HF_ORG}/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    done
  done
done


#################
# TASK SPECIFIC #
#################
# mt5: NEED TO INSTALL `lmqg` https://github.com/asahi417/lm-question-generation
HF_ORG='vocabtrimmer'
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  for SIZE in "small" "base"
  do
    for TARGET in 15000 30000 45000 60000 75000 90000 120000 150000 180000 210000
    do
      MODEL="mt5-${SIZE}-${LA}quad-qg-trimmed-${TARGET}"
      vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "${HF_ORG}/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
      git clone "https://huggingface.co/${HF_ORG}/${MODEL}"
      rm -rf "${MODEL}/eval"
      lmqg-eval -m "${MODEL}" -e "${MODEL}/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
      cd "${MODEL}" && git add . && git commit -m "add eval" && git push && cd ..
      rm -rf "${MODEL}"
    done
  done
done
