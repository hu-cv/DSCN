function results=run_DCSN(seq, res_path, bSaveImage)

close all;
results = DCSN(seq.path, seq.ext, false, seq.init_rect, seq.startFrame,seq.endFrame);
end
