reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 853+854+855+856+857+858+859
select dfg_inactive, inactive and resi 853+854+855+856+857+858+859
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 755+756+757+758+759+760+761+762+763+764+765+766+767+768+769
select alphac_inactive, inactive and resi 755+756+757+758+759+760+761+762+763+764+765+766+767+768+769
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 855+856+857+858+859+860+861+862+863+864+865+866+867+868+869+870+871+872+873+874+875+876+877+878+879
select activation_loop_inactive, inactive and resi 855+856+857+858+859+860+861+862+863+864+865+866+867+868+869+870+871+872+873+874+875+876+877+878+879
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 833+834+835+836+837+838+839
select hrd_inactive, inactive and resi 833+834+835+836+837+838+839
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 745+762
select lys_glu_inactive, inactive and resi 745+762
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
