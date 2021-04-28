
for i in {0..9}; do
    CLUSTER=$i
    CLS="../HISourceFinder/Test_data/Input/noisefree_${CLUSTER}.fits"
    LBL="../HISourceFinder/Test_data/Target/mask_$(echo $CLS | cut -d'_' -f 3)"
    echo "Cube: ${CLS}"
    echo "Ground-truth: ${LBL}"

    MTO_BASE="res/c${CLUSTER}_mto"
    MTO_FITS="${MTO_BASE}.fits"
    MTO_PLOT="${MTO_BASE}.png"
    MTO_LOGS="${MTO_BASE}.log"

    echo "${MTO_LOGS}"
    MTO_SCORE="${MTO_BASE}_score.csv"
    MTO_SCORE_PLOT="${MTO_BASE}_score.png"

    if [[ ! -f $MTO_FITS ]]; then
        echo -n "Classifying with MTO ... "
        ./mto-lvq 1 16 "$CLS" 1 "" "" 32 1 "segcube_$CLUSTER" 1 0.0016 0 > "$MTO_LOGS"
        mv segmcube.fits "$MTO_FITS"
        echo "done!"
    fi
    LVQ="cluster${CLUSTER}_mto.joblib"
    echo "Classifier: ${LVQ}"

    LVQ_BASE="res/c${CLUSTER}_mto_lvq"
    LVQ_FITS="${LVQ_BASE}.fits"
    LVQ_PLOT="${LVQ_BASE}.png"
    LVQ_LOGS="${LVQ_BASE}.log"
    LVQ_LOGS_2="${LVQ_BASE}_training.log"

    LVQ_SCORE="${LVQ_BASE}_score.csv"
    LVQ_SCORE_PLOT="${LVQ_BASE}_score.png"

    if [[ ! -f "$LVQ" ]]; then
        echo -n "Training LVQ classifier..."
        ./mto-lvq 1 16 "$CLS" 1 "$LVQ" "$MTO_FITS" 32 1 "$CLASSIFIER$CLUSTER" 1 #0.0016 0 > "$LVQ_LOGS_2"
        echo "done!"
    fi

    if [[ ! -f "$LVQ_FITS" ]]; then
        echo -n "Classifying with MTO+LVQ ... "
        ./mto-lvq 1 16 "$CLS" 1 "$LVQ" "" 32 1 "$CLASSIFIER$CLUSTER" 1 0.0016 0 > "$LVQ_LOGS"
        mv segmcube.fits "$LVQ_FITS"
        echo "done!"
    fi
    CLUSTER=$(( CLUSTER + 1 ))
    echo index $CLUSTER
done
