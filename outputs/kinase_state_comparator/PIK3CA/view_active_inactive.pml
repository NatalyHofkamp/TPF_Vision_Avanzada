reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 931+932+933+934+935+936+937
select dfg_inactive, inactive and resi 931+932+933+934+935+936+937
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 58+59+60+61+62+63+64+65+66+67+68+69+70+71+72
select alphac_inactive, inactive and resi 58+59+60+61+62+63+64+65+66+67+68+69+70+71+72
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 933+934+935+936+937+938+939+940+941+942+943+944+945+946+947+948+949+950+951+952+953+954+955+956+957
select activation_loop_inactive, inactive and resi 933+934+935+936+937+938+939+940+950+951+952+953+954+955+956+957+958+959+960+961+962+963+964+965+966
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select lys_glu_active, active and resi 46+65
select lys_glu_inactive, inactive and resi 46+65
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
