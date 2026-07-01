#!/bin/bash 

if [[ $# -lt 2 ]]; then echo $0 [ACCESSION NUMBER] [OUTPUT DIRECTORY] ; exit; fi

export accession=$1
export output_dir=$2

mkdir -vp $output_dir/STUDY_INFO

#Searches dcmdump for value of specific tag
get_tag() { dcmdump +P "$1" "$2" | grep -o -P '(?<=\[)(.*?)(?=\])'; }

#Sorts DICOMs by PatientID, Accession Number, and Series 
sortd() { 
patient=`get_tag PatientID $1`
series=`get_tag SeriesDescription $1` 
seriesnum=`get_tag SeriesNumber $1`
dpath="$output_dir"/"$patient"/"$accession"/"$seriesnum"_"$series"
if [[ ! "$patient" == "" ]] && [[ ! "$accession" == "" ]] && [[ ! "$seriesnum" == "" ]] && [[ ! "$series" == "" ]] ; then 
 if [[ ! -d "$dpath" ]]; then
  mkdir -vp "$dpath"
  mv --backup=t "$1" "$dpath"
 else
  mv --backup=t "$1" "$dpath"
 fi
fi;
}

export -f get_tag sortd

findscu -od $output_dir/STUDY_INFO -X +sr -aet RESEARCHPACS -aec PACSDCM -S -k "QueryRetrieveLevel=STUDY" -k "AccessionNumber=$accession" -k StudyInstanceUID  pacsstor.tch.harvard.edu 104
for d in $output_dir/STUDY_INFO/rsp*.dcm; do 
	getscu -od $output_dir -aet RESEARCHPACS -aec PACSDCM -S pacsstor.tch.harvard.edu 104 $d
done 

find "$output_dir" -maxdepth 1 -type f -print | parallel -j `nproc` -k sortd
