SRC=$1
DST=$2

OLD_NAME=$(basename $SRC)
NEW_NAME=$(basename $DST)

echo $OLD_NAME
echo $NEW_NAME
mkdir -p $DST

cp -r $SRC/* $DST/

for f in $SRC/*
do
    sed "s/${OLD_NAME}/${NEW_NAME}/g" $f > $DST/$(basename $f)
done