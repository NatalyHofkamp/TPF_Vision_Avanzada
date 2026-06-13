reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 861+862+863+864+865+866+867
select dfg_inactive, inactive and resi 861+862+863+864+865+884+885
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 763+764+765+766+767+768+769+770+771+772+773+774+775+776+777
select alphac_inactive, inactive and resi 763+764+765+766+767+768+769+770+771+772+773+774+775+776+777
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 863+864+865+866+867+868+869+870+871+872+873+874+875+876+877+878+879+882+883+884+885+886+887+888+889
select activation_loop_inactive, inactive and resi 863+864+865+884+885+886+887+888+889+890+891+892+893+894+895+896+897+898+899+900+901+902+903+904+905
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 841+842+843+844+845+846+847
select hrd_inactive, inactive and resi 841+842+843+844+845+846+847
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 753+770
select lys_glu_inactive, inactive and resi 753+770
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
