function [valid_x,valid_y]=extract_minutiae(I)

    ext_window=3; 
    ext_margin=7;    
    val_window=3;

    [~,~,~,~,~,enhI] =  fft_enhance_cubs(I, -1);
    [~, binI,~,~, I1_enhaced] =  testfin(enhI);
    inv_binI = (binI == 0); 
    thin =  bwmorph(inv_binI, 'thin',Inf);
    [minutiae, minutiae_x, minutiae_y,~]=extraction(thin,ext_window,ext_margin);
    [valid, valid_x, valid_y,~]=validation(thin,minutiae,val_window);

end