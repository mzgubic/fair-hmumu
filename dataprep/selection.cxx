void selection(std::string infile, std::string outfile, int full_selection, int is_signal) {

  std::cout << full_selection << std::endl << infile << std::endl << outfile << std::endl;

  //::: Get the old file, and fetch the old tree
  TFile* in_file = new TFile(infile.c_str());
  std::string tree_name = ( full_selection ) ? "DiMuonNtuple" : "di_muon_ntuple";
  TTree* in_tree = (TTree*)in_file->Get(tree_name.c_str());
  Long64_t nentries = in_tree->GetEntries();

  Float_t dimuon_mass = 0;
  Float_t leadeta = 0;
  Float_t subeta = 0;
  Float_t met = 0;
  Float_t leadpt = 0;
  Float_t subpt = 0;
  Float_t zpt = 0;
  Int_t hasbjet = 0;
  Int_t nmuons = 0;
  in_tree->SetBranchAddress("Muons_Minv_MuMu", &dimuon_mass);
  in_tree->SetBranchAddress("Muons_Eta_Lead", &leadeta);
  in_tree->SetBranchAddress("Muons_Eta_Sub", &subeta);
  in_tree->SetBranchAddress("Muons_PT_Lead", &leadpt);
  in_tree->SetBranchAddress("Muons_PT_Sub", &subpt);
  in_tree->SetBranchAddress("Z_PT", &zpt);
  if ( full_selection ) {
    in_tree->SetBranchAddress("Event_MET", &met);
    in_tree->SetBranchAddress("Event_HasBJet", &hasbjet);
    in_tree->SetBranchAddress("Muons_Multip", &nmuons);
  }

  // Do not copy the majority of the branches
  in_tree->SetBranchStatus("*", 0);

  // Turn these branches on
  in_tree->SetBranchStatus("EventInfo_EventNumber", 1);
  in_tree->SetBranchStatus("Muons_PT_Lead", 1);
  in_tree->SetBranchStatus("Muons_PT_Sub", 1);
  in_tree->SetBranchStatus("Muons_Eta_Lead", 1);
  in_tree->SetBranchStatus("Muons_Eta_Sub", 1);
  in_tree->SetBranchStatus("Muons_Phi_Lead", 1);
  in_tree->SetBranchStatus("Muons_Phi_Sub", 1);
  in_tree->SetBranchStatus("Z_PT", 1);
  in_tree->SetBranchStatus("Z_Eta", 1);
  in_tree->SetBranchStatus("Z_Phi", 1);
  in_tree->SetBranchStatus("Z_Y", 1);
  in_tree->SetBranchStatus("Muons_Minv_MuMu", 1);
  in_tree->SetBranchStatus("Muons_CosThetaStar", 1);
  in_tree->SetBranchStatus("Muons_DeltaEta_MuMu", 1);
  in_tree->SetBranchStatus("Muons_DeltaPhi_MuMu", 1);
  in_tree->SetBranchStatus("Muons_DeltaR_MuMu", 1);
  in_tree->SetBranchStatus("Jets_E_Lead", 1);
  in_tree->SetBranchStatus("Jets_E_Sub", 1);
  in_tree->SetBranchStatus("Jets_PT_Lead", 1);
  in_tree->SetBranchStatus("Jets_PT_Sub", 1);
  in_tree->SetBranchStatus("Jets_Eta_Lead", 1);
  in_tree->SetBranchStatus("Jets_Eta_Sub", 1);
  in_tree->SetBranchStatus("Jets_Phi_Lead", 1);
  in_tree->SetBranchStatus("Jets_Phi_Sub", 1);
  in_tree->SetBranchStatus("Jets_PT_jj", 1);
  in_tree->SetBranchStatus("Jets_Minv_jj", 1);
  in_tree->SetBranchStatus("Jets_etaj1_x_etaj2", 1);
  in_tree->SetBranchStatus("Jets_DeltaEta_jj", 1);
  in_tree->SetBranchStatus("Jets_DeltaPhi_jj", 1);
  in_tree->SetBranchStatus("Jets_DeltaR_jj", 1);
  if ( full_selection ) { // copy for analysis ntuples, change to Int_t for SS ntuples
    in_tree->SetBranchStatus("Jets_jetMultip", 1);
  }

  //::: Prepare the new tree
  TFile *out_file = new TFile(outfile.c_str(), "recreate");
  TTree *out_tree = in_tree->CloneTree(0);

  //::: Set the same name for all trees
  out_tree->SetName("DiMuonNtuple");

  //::: Copy the weight and rename ExpWeight -> GlobalWeight. Branch rename is not possible, so:
  //::: Turn on branch after Clone(), it shouldn't be copied, but we want access to the value.
  std::string weight_name = ( full_selection ) ? "GlobalWeight" : "ExpWeight";
  in_tree->SetBranchStatus(weight_name.c_str(), 1);
  Float_t event_weight = 0;
  in_tree->SetBranchAddress( weight_name.c_str(), &event_weight);
  out_tree->Branch("GlobalWeight", &event_weight, "GlobalWeight/F");

  //::: Similarly spurious signal files have jet multiplicity stored as a float: change to int
  Int_t njets_Int = 0;
  Float_t njets_Float = 0;
  if ( !full_selection ) {
    in_tree->SetBranchStatus("Jets_jetMultip", 1);
    in_tree->SetBranchAddress("Jets_jetMultip", &njets_Float);
    out_tree->Branch("Jets_jetMultip", &njets_Int, "Jets_jetMultip/I");
  }

  //::: Add new branches (Wisconsin)
  Float_t leadptm, subptm, zptm;
  out_tree->Branch("Muons_PTM_Lead", &leadptm, "Muons_PTM_Lead/F");
  out_tree->Branch("Muons_PTM_Sub", &subptm, "Muons_PTM_Sub/F");
  out_tree->Branch("Z_PTM", &zptm, "Z_PTM/F");

  //::: Add a new column, IsSignal
  out_tree->Branch("IsSignal", &is_signal, "IsSignal/I");

  //::: Loop over the tree, apply the cuts 
  for (Long64_t i=0; i<nentries; i++) {
    in_tree->GetEntry(i);
    if ( dimuon_mass > 110 && dimuon_mass < 160 ) {
      if ( abs(leadeta) < 2.5 && abs(subeta) < 2.5 ) {

        // compute the new variables
        leadptm = leadpt/dimuon_mass;
        subptm = subpt/dimuon_mass;
        zptm = zpt/dimuon_mass;

        // apply full selection
        if ( full_selection ) {
          if ( hasbjet == 0 && nmuons < 3 && met < 80 ) {
            out_tree->Fill();
          }
        }
        // or not
        else {
          njets_Int = njets_Float;
          out_tree->Fill();
        }

      }
    }
  } // loop over the tree

  //::: Clean up
  out_tree->AutoSave();
  delete in_file;
  delete out_file;

}
