SherpaVersion="2.2.5"
HepMCVersion="2.06.09"
RivetVersion="2.5.0"  # 2.6.0 not stable
FastJetVersion="3.2.2"
LHAPDFVERSION="6.1.6"
rootVersion="6.02.08"
#MCFMVersion="8.0"

curDir=$PWD

SherpaDir=$curDir"/"$SherpaVersion
cd ${SherpaDir}

HEPTOOLS_DIR="/home/chrbecker/HEP-Tools"
#RivetDir=${SherpaDir}/Rivet-${RivetVersion}

#HepMC=${SherpaDir}/HepMC-${HepMCVersion}
#Rivet=${SherpaDir}/Rivet-${RivetVersion}/Rivet
Rivet=${HEPTOOLS_DIR}"/rivet/"${RivetVersion}
#HepMC=${HEPTOOLS_DIR}"/HepMC/"$HepMCVersion
HepMC=${Rivet}
#FastJet=${HEPTOOLS_DIR}"/FastJet/"${FastJetVersion}
FastJet=${Rivet}
LHAPDF=${HEPTOOLS_DIR}"/LHAPDF/"${LHAPDFVERSION}
OpenLoops=${HEPTOOLS_DIR}"/OpenLoops"
Root="/cvmfs/pheno.egi.eu/HEJ/Dependencies/root-"${rootVersion}

echo "$HepMC"
echo "$Rivet"
echo "$FastJet"
echo "$LHAPDF"
echo "$Root"
#echo "$MCFM"
mkdir -p $SherpaDir
cd $SherpaDir
#wget http://www.hepforge.org/archive/sherpa/SHERPA-MC-${SherpaVersion}.tar.gz
#tar -zxf SHERPA-MC-${SherpaVersion}.tar.gz

git clone -b rel-2-2-5 https://gitlab.com/sherpa-team/sherpa.git SHERPA-MC-${SherpaVersion}
cd SHERPA-MC-${SherpaVersion}
echo "
------------------------------------------------------------------------
|                              BEGIN Configure                          |
------------------------------------------------------------------------
"

#cd ${SherpaVersion}
autoreconf -i
./configure --prefix ${SherpaDir} --enable-hepmc2=${HepMC} --enable-rivet=${Rivet} --enable-fastjet=${FastJet} --enable-lhapdf=${LHAPDF} CXXFLAGS="-std=c++11" --enable-mpi --enable-openloops=${OpenLoops} | tee configure.log
#--enable-ufo --enable-root=${Root} --enable-gzip

echo "
------------------------------------------------------------------------
|                              BEGIN Install                          |
------------------------------------------------------------------------
"
#make -j6
#make install

cd ..

#rm -f SHERPA-MC-${SherpaVersion}.tar.gz
