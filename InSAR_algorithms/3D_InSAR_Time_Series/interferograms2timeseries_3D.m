function [time_series time_series1]=interferograms2timeseries_3D_new(data,theta1,alfa1,N_igram1,before1,after1,data1)
% function time_series = interferograms2timeseries_3D_new(data,theta1,alfa1,N_igram1,before1,after1,data1)

%
% Performs inversion and substract obtained synth. interf. from ori. interf.
%
% [time_series]=interferograms2timeseries(data)
%
%
%       data      :         Igram structure with filtered unwrapped interferograms
%
%
% W.Zhao Nov. 2011
%

%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%

dim_data = length(data);   
for ni = 1:dim_data
    data(ni).data = single(data(ni).data);
end
[dayslist1,datelist1] = CalcDateList(data1);
for ni=N_igram1+1:dim_data
%     data(ni).t1 = data(ni).t1 + dayslist1(5) + 1;
%     data(ni).t2 = data(ni).t2 + dayslist1(5) + 1;
    data(ni).t1 = data(ni).t1 + dayslist1(5);      %%%% for merge igram1 & igram2 mode, don't plus 1
    data(ni).t2 = data(ni).t2 + dayslist1(5);
end
clear data1


% %%%%% remove interferogram for model resolution
% [dayslist,datelist] = CalcDateList(data);
% day_rm = [7 15 20 23 29 30 31 34 35 37 43 46 48 54];
% data_bak = data;
% for ni = 1:dim_data
%     for nj = 1:length(day_rm)
%         if data(ni).t1 == dayslist(day_rm(nj)) || data(ni).t2 == dayslist(day_rm(1,nj))
%             data_bak(ni).clear = 1;
%         else
%             data_bak(ni).clear = 0;
%         end
%     end
% end
% data = [];
% for ni =1:dim_data
%     if data_bak(ni).clear ==0
%         data = [data data_bak(ni)];
%     end
% end
% clear data_bak;  clear dayslist;   clear datelist
% dim_data = size(data,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





[dayslist,datelist] = CalcDateList(data);     dt = diff(dayslist);
M = Make_designMatrix_3D(data,before1,after1);  dim_model = size(M,2);
line = size(data(1).data,1);
col = size(data(1).data,2);

igram_temp = data(1);
vec = [63.998 -18.273];
[outind]=LL2ind_igram(igram_temp,vec);
col1 = outind(2);
line1 = outind(1);

for ni =2:dim_data
    data(1).data(isnan(data(ni).data))=nan;
end
for ni = 1:dim_data
    data(ni).data(isnan(data(1).data))=nan;
    temp1 = data(ni).data(line1-1:line1+1,col1-1:col1+1);
    temp2 = find(~isnan(temp1));
    k = 1;
    p = 1;
    while k==1
        p = p + 1;
        if length(temp2) == 0
            temp1 = inData(line1-p:line1-p,col1-p:col1+p);
            temp2 = find(~isnan(temp1));
        else
            k = 0;
            temp1(isnan(temp1)) = 0;
            ME = sum(sum(temp1))/length(temp2);
        end
    end
        data(ni).data = data(ni).data - ME;
%     data(ni).data = reshape(data(ni).data,line*col,1);
end

tmpdata = single(zeros(dim_data,size(data(1).data,1),size(data(1).data,2))) ;
datamat = single(zeros(dim_data,size(data(1).data,1)*size(data(1).data,2))) ;

for ni=1:dim_data ;  tmpdata(ni,:,:) = data(ni).data;  end

datamat(:) = tmpdata;  clear tmpdata;
nb_pixel   = numel(data(1).data) ;  %tmpe_defo = single(zeros(dim_model,nb_pixel)) ;
% velo_v = single(zeros(length(dayslist)-before1-after1,nb_pixel));
% velo_e = single(zeros(length(dayslist)-before1-after1,nb_pixel));
def_v = single(zeros(length(dayslist)-before1-after1+1,nb_pixel));
def_e = single(zeros(length(dayslist)-before1-after1+1,nb_pixel));
dt1 = dt(before1+1:length(dayslist)-after1);
for ki=1:nb_pixel
    temp_data = datamat(:,ki);
    if isnan(temp_data(1))
        def_v(:,ki) = def_v(:,ki).*NaN;%tmpe_defo(:,ki) = defo;
        def_e(:,ki) = def_e(:,ki).*NaN;
    elseif length(find(isnan(theta1(:,ki)))) || length(find(isnan(alfa1(:,ki))))
        def_v(:,ki) = def_v(:,ki).*NaN;
        def_e(:,ki) = def_e(:,ki).*NaN;
    else
        M1 = ones(dim_data,dim_model);  
        N_temp1 = length(before1+1:2:dim_model-after1+1);   N_temp2 = length(before1+2:2:dim_model-after1+1);
        angle1_base = cosd(theta1(:,ki));   angle2_base = sind(theta1(:,ki)).*cosd(alfa1(:,ki));
        angle1 = repmat(angle1_base,1,N_temp1);   angle2 = repmat(angle2_base,1,N_temp2);
        M1(:,before1+1:2:dim_model-after1+1) = angle1;  M1(:,before1+2:2:dim_model-after1+1) = angle2;
        M2 = M1.*M;
        M3 = pinv(M2);
        DR = M2*M3;
        W = abs(diag(DR)).^6;
        M4 = repmat(W,1,dim_model);
        M5 = M2.*M4;
        M6 = pinv(M5);
        temp_data1 = temp_data.*W;
        tmp_rate = M6*temp_data1;
        tmp_v = tmp_rate(before1+1:2:dim_model-after1);      %velo_v(:,ki) = tmp_v;
        defo1        = cumsum([0 ; tmp_v.*dt1]);             def_v(:,ki) = defo1;
        tmp_e = tmp_rate(before1+2:2:dim_model-after1+1);      %velo_e(:,ki) = tmp_v;
        defo2        = cumsum([0 ; tmp_e.*dt1]);             def_e(:,ki) = defo2;
    end
    if (~rem(ki,20000)) disp(sprintf('Processing Line: %5d ',ki)) ;  end
end


for ni=1:size(def_v,1)
    tmpigram = rmfield(data(1),{'file_length' 'width' 'date1' 'date2' 't1' 't2' 'delt'})    ;    % 'bperptop' 'bperpbot' remains in the structure when roi_pac interfero,need to deal with
    rad2disp            = convert_unit('m',data,data(1).wavelength)                                               ;
    tmpigram.data  = reshape(def_v(ni,:,:), line,col) * rad2disp         ;   % Transform unit to meters
    tmpigram.width      = col ;  tmpigram.file_length = line                     ;
    tmpigram.t          = dayslist(ni+before1) - dayslist(before1+1)         ;  tmpigram.date        = datelist(ni+before1,:)                           ;
    tmpigram.date_years = yymmdd2y(datelist(ni+before1,:)) ;   tmpigram.Unit = 'm' ;                                                            
    if isfield(data(1),'wavelength')      tmpigram.wavelength=data(1).wavelength;  else  tmpigram.wavelength=5.56 ;  end
    time_series(ni)     = tmpigram;
    
    tmpigram1 = rmfield(data(1),{'file_length' 'width' 'date1' 'date2' 't1' 't2' 'delt'})    ;    % 'bperptop' 'bperpbot' remains in the structure when roi_pacinterfero,need to deal with
    rad2disp            = convert_unit('m',data,data(1).wavelength)                                               ;
    tmpigram1.data  = reshape(def_e(ni,:,:), line,col) * rad2disp         ;   % Transform unit to meters
    tmpigram1.width      = col ;  tmpigram1.file_length = line                     ;
    tmpigram1.t          = dayslist(ni+before1) - dayslist(before1+1)         ;  tmpigram1.date        = datelist(ni+before1,:)                           ;
    tmpigram1.date_years = yymmdd2y(datelist(ni+before1,:)) ;   tmpigram1.Unit = 'm' ;                                                            ;
    if isfield(data(1),'wavelength')      tmpigram1.wavelength=data(1).wavelength;  else  tmpigram1.wavelength=5.56 ;  end
    time_series1(ni)     = tmpigram1;
end


