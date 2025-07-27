import streamlit as st
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from stmol import showmol
import py3Dmol
from rdkit.Chem import AllChem

def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view()    #(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview,height=500,width=500)
    
def calculate_descriptors(smile, descriptor_name):
    molecule = Chem.MolFromSmiles(smile)
    descriptor_func = getattr(Descriptors, descriptor_name)
    data = descriptor_func(molecule)
    return data

def predict(smiles, ip=None, homo_gap=None):

    if ip != None and homo_gap != None:
        lf_fea = ["HomoLumoGap","Chi2n","MolMR","Chi1n","BCUT2D_LOGPLOW","IP","AvgIpc","MaxPartialCharge","BCUT2D_MWHI","fr_halogen"]
        gwp_fea = ["IP","qed","FractionCSP3","SMR_VSA7","MinAbsEStateIndex","FpDensityMorgan2","BCUT2D_LOGPHI","Chi2v","HomoLumoGap","Chi0v"]
    
        lf_x = []
        for fea in lf_fea:
            if fea == "HomoLumoGap":
                f = homo_gap
            elif fea == "IP":
                f = ip
            else:
                f = calculate_descriptors(smiles, fea)
            lf_x.append(f)
    
        gwp_x = []
        for fea in gwp_fea:
            if fea == "HomoLumoGap":
                f = homo_gap
            elif fea == "IP":
                f = ip
            else:
                f = calculate_descriptors(smiles, fea)
            gwp_x.append(f)
        
        scaler = joblib.load("./webapp/scaler_MLP_LF.gz")
        lf_x = scaler.transform(np.array(lf_x).reshape(1, -1))
        
        model_lf = joblib.load('./webapp/mlp_LF.pkl')
        model_gwp = joblib.load('./webapp/rf_GWP.pkl')
    
        lifetime = 10 ** model_lf.predict(lf_x)
        gwp = 10 ** model_gwp.predict(np.array(gwp_x).reshape(1, -1))
    else:
        lf_fea = ["Chi2n","MolMR","Chi1n","BCUT2D_LOGPLOW","AvgIpc","MaxPartialCharge","BCUT2D_MWHI","fr_halogen"]
        gwp_fea = ["qed","FractionCSP3","SMR_VSA7","MinAbsEStateIndex","FpDensityMorgan2","BCUT2D_LOGPHI","Chi2v","Chi0v"]
        lf_x = []
        for fea in lf_fea:
            f = calculate_descriptors(smiles, fea)
            lf_x.append(f)
    
        gwp_x = []
        for fea in gwp_fea:
            f = calculate_descriptors(smiles, fea)
            gwp_x.append(f)
    
        scaler = joblib.load("./webapp/scaler_MLP_LF_geo.gz")
        lf_x = scaler.transform(np.array(lf_x).reshape(1, -1))
        
        model_lf = joblib.load('./webapp/mlp_LF_geo.pkl')
        model_gwp = joblib.load('./webapp/rf_GWP_geo.pkl')
    
        lifetime = 10 ** model_lf.predict(lf_x)
        gwp = 10 ** model_gwp.predict(np.array(gwp_x).reshape(1, -1))
        
    return float(gwp), float(lifetime)

st.title('Global Warming Potential and Atmospheric Lifetime Prediction')
st.markdown(" :alien: **GWP predicted By Random Forest & AL predicted By Multi-layer Perceptron**")                
st.markdown(" :tired_face: If IP and HOMO-LUMO Gap are inputted, then the prediction will be more accurate")                   
st.markdown("Contact: sxmzhaogb@gmail.com")
st.markdown(' :heart_eyes: <span style="color:grey;">Cite as: G. Zhao, H. Kim, C. Ynag, Y. G. Chung. Leveraging Machine Learning to Predict the Atmospheric Lifetime and the Global Warming Potential (GWP) of SF6 Replacement Gases. DOI: 10.1021/acs.jpca.3c07339 </span>', unsafe_allow_html=True)
# model_type = st.selectbox('Choose a model type:', ['GBR', 'MLP'])
smiles = st.text_input('Enter the SMILES representation:','C')
blk=makeblock(smiles)
render_mol(blk)

ip_input = st.text_input('Enter the Ionization Potential (IP):')
homo_gap_input = st.text_input('Enter the HOMO-LUMO Gap:')

if st.button('Predict'):
    ip = homo_gap = None
    error = False
    if ip_input:
        try:
            ip = float(ip_input)
        except ValueError:
            st.error('Ionization Potential (IP) must be a valid number.')
            error = True
    if homo_gap_input:
        try:
            homo_gap = float(homo_gap_input)
        except ValueError:
            st.error('HOMO-LUMO Gap must be a valid number.')
            error = True
    if not error and smiles:
        gwp, lifetime = predict(smiles, ip, homo_gap)
        st.write(f'Predicted Global Warming Potential: {gwp:.4f}')
        st.write(f'Predicted Atmospheric Lifetime: {lifetime:.4f} year.')
    elif not smiles:
        st.error('Please enter the SMILES representation.')


st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao, Haewon Kim and Prof. Yongchul G. Chung (Pusan National University)</span>', unsafe_allow_html=True)
