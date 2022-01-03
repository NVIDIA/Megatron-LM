for i in {12..14}
do
  let imin=${i}-1
  echo sed -i "s/pm0${imin}/pm0${i}/g" preprocess.lsf
  sed -i "s/pm0${imin}/pm0${i}/g" preprocess.lsf
  bsub preprocess.lsf
  
done

