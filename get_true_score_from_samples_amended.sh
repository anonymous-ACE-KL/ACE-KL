for ((i=0; i<4; i++))
do
    ./hmnb_pred 5HMNB1 data/bad${i}_amended.dat data/bad${i}_amended.scores 2
done

for ((i=0; i<4; i++))
do
    ./hmnb_pred 5HMNB1 data/bad${i}_lamended.dat data/bad${i}_random_lamended.scores 2
done

for ((i=0; i<4; i++))
do
    ./hmnb_pred 5HMNB1 data/bad${i}_random_amended.dat data/bad${i}_random_amended.scores 2
done

