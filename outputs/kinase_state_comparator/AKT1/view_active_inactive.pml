reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 290+291+292+293+294+295+296
select dfg_inactive, inactive and resi 290+291+292+293+294+295+296
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 177+178+179+180+181+182+183+184+185+186+187+188+189+190+191
select alphac_inactive, inactive and resi 177+178+179+180+181+182+183+184+185+186+204+205+206+207+208
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 292+293+294+295+296+297+298+299+300+301+302+303+304+305+306+307+309+310+311+312+313+314+315+316+317
select activation_loop_inactive, inactive and resi 292+293+294+295+296+297+298+299+300+301+309+310+311+312+313+314+315+316+317+318+319+320+321+322+323
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select lys_glu_active, active and resi 168+184
select lys_glu_inactive, inactive and resi 168+184
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
