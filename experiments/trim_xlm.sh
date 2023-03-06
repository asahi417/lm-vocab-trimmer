#####################
# FOUNDATION MODELS #
#####################
# xlm-r
for SIZE in "base" "large"
do
  for LA in 'fr' 'de' 'es' 'it' 'ar' 'pt'
  do
    MODEL="xlm-roberta-${SIZE}-trimmed-${LA}"
    vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}"
    rm -rf "ckpts/${MODEL}"
  done
done


for SIZE in "base" "large"
do
  for LA in 'fr' 'de' 'es' 'it' 'ar' 'pt'
  do
    for TARGET in 15000 30000 45000 60000
    do
      MODEL="xlm-roberta-${SIZE}-trimmed-${LA}-${TARGET}"
      vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
      rm -rf "ckpts/${MODEL}"
    done
  done
done

for SIZE in "base" "large"
do
  for LA in 'fr' 'de' 'es' 'it' 'pt'
  do
    TARGET=75000
    MODEL="xlm-roberta-${SIZE}-trimmed-${LA}-${TARGET}"
    vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
    rm -rf "ckpts/${MODEL}"
  done
done


for SIZE in "base" "large"
do
  for LA in 'fr' 'de' 'es'
  do
    for TARGET in 90000 105000
    do
      MODEL="xlm-roberta-${SIZE}-trimmed-${LA}-${TARGET}"
      vocabtrimmer-trimming -m "xlm-roberta-${SIZE}" -l "${LA}" --repo-id "vocabtrimmer/${MODEL}" -p "ckpts/${MODEL}" -v ${TARGET}
      rm -rf "ckpts/${MODEL}"
    done
  done
done