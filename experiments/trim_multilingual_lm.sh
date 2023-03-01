HF_ORG='asahi417'

#####################
# FOUNDATION MODELS #
#####################
# mt5
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "google/mt5-small" -l "${LA}" --repo-id "${HF_ORG}/mt5-small-trimmed-${LA}"
  vocabtrimmer-trimming -m "google/mt5-base" -l "${LA}" --repo-id "${HF_ORG}/mt5-base-trimmed-${LA}"
done

# xlm-r
for LA in 'ja' 'fr' 'de' 'es' 'it' 'ar' 'pt'
do
  vocabtrimmer-trimming -m "xlm-roberta-base" -l "${LA}" --repo-id "${HF_ORG}/xlm-roberta-base-trimmed-${LA}"
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
    for TARGET in 15000 30000 45000 60000 75000
    do
      vocabtrimmer-trimming -m "lmqg/mt5-${SIZE}-${LA}quad-qg" -l "${LA}" --repo-id "${HF_ORG}/mt5-${SIZE}-${LA}quad-qg-trimmed-${TARGET}" -p "ckpts/mt5-${SIZE}-${LA}quad-qg-trimmed" -v ${TARGET}
      git clone "https://huggingface.co/${HF_ORG}/mt5-${SIZE}-${LA}quad-qg-trimmed"
      rm -rf "mt5-${SIZE}-${LA}quad-qg-trimmed/eval"
      lmqg-eval -m "mt5-${SIZE}-${LA}quad-qg-trimmed" -e "mt5-${SIZE}-${LA}quad-qg-trimmed/eval" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
      cd "mt5-${SIZE}-${LA}quad-qg-trimmed" && git add . && git commit -m "add eval" && git push && cd ..
      rm -rf "mt5-${SIZE}-${LA}quad-qg-trimmed"
    done
  done
done


########
# MISC #
########
# mbart
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
  vocabtrimmer-trimming -m "facebook/mbart-large-cc25" -l "${LA}" --repo-id "${HF_ORG}/mbart-large-cc25-trimmed-${LA}"
done

# mbert
for LA in 'ja' 'fr' 'de' 'es' 'it' 'ar' 'pt'
do
  vocabtrimmer-trimming -m 'bert-base-multilingual-cased' -l "${LA}" --repo-id "${HF_ORG}/'bert-base-multilingual-cased-trimmed-${LA}"
done
