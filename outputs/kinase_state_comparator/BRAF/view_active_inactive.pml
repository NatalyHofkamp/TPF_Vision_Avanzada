reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 592+593+594+595+596+597+598
select dfg_inactive, inactive and resi 592+593+594+595+596+597+598
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 494+495+496+497+498+499+500+501+502+503+504+505+506+507+508
select alphac_inactive, inactive and resi 494+495+496+497+498+499+500+501+502+503+504+505+506+507+508
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 594+595+596+597+598+599+600+601+602+603+610+611+612+613+614+615+616+617+618+619+620+621+622+623+624
select activation_loop_inactive, inactive and resi 594+595+596+597+598+599+600+601+602+603+604+605+606+607+608+609+610+611+612+613+614+615+616+617+618
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 572+573+574+575+576+577+578
select hrd_inactive, inactive and resi 572+573+574+575+576+577+578
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 483+501
select lys_glu_inactive, inactive and resi 483+501
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
