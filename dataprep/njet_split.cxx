void njet_split(std::string infile, std::string ofile ) {

  //::: Get the old file, and fetch the old tree
  TFile* in_file = new TFile(infile.c_str());
  TTree* in_tree = (TTree*)in_file->Get( "DiMuonNtuple" );
  Long64_t nentries = in_tree->GetEntries();

  //::: Register njets
  Int_t njets = 0;
  in_tree->SetBranchAddress("Jets_jetMultip", &njets);

  //::: Copy all
  in_tree->SetBranchStatus("*", 1);

  //::: Prepare the new trees
  TFile *out_file = new TFile(ofile.c_str(), "recreate");
  TTree *out_tree_0 = in_tree->CloneTree(0);
  TTree *out_tree_1 = in_tree->CloneTree(0);
  TTree *out_tree_2 = in_tree->CloneTree(0);
  out_tree_0->SetName("0jet");
  out_tree_1->SetName("1jet");
  out_tree_2->SetName("2jet");

  //::: Loop over the tree, apply the cuts 
  for (Long64_t i=0; i<nentries; i++) {

    in_tree->GetEntry(i);

    // see how many jets there are
    if ( njets == 0 ) {
      out_tree_0->Fill();
    }
    else if ( njets == 1 ) {
      out_tree_1->Fill();
    }
    else if ( njets >= 2 ) {
      out_tree_2->Fill();
    }

  }

  //::: Clean up
  out_tree_0->AutoSave();
  out_tree_1->AutoSave();
  out_tree_2->AutoSave();
  delete in_file;
  delete out_file;

}
