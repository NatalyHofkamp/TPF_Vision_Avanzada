reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 808+809+810+811+812+813+814
select dfg_inactive, inactive and resi 808+809+810+811+812+813+814
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 633+634+635+636+637+638+639+640+641+642+643+644+645+646+647
select alphac_inactive, inactive and resi 633+634+635+636+637+638+639+640+641+642+643+644+645+646+647
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 810+811+812+813+814+815+816+817+818+819+820+821+822+823+824+825+826+827+828+829+830+831+832+833+834
select activation_loop_inactive, inactive and resi 810+811+812+813+814+815+816+817+818+819+820+821+822+823+824+825+826+827+828+829+830+831+832+833+834
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 788+789+790+791+792+793+794
select hrd_inactive, inactive and resi 788+789+790+791+792+793+794
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 623+640
select lys_glu_inactive, inactive and resi 623+640
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
