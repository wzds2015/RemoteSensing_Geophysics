//
//  main.cpp
//  cpxlook
//
//  Created by Admin on 5/27/14.
//  Copyright (c) 2014 Wenliang Zhao. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <NUMERIC>
#include <math.h>
#include "file_tools.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "float complex interferogram multi-looking. Syntax = cpxlook $inputfile $width $looks $method(mean or median) $outputfile.\n";
    cout << "Processing " << argv[1] << ", number of look: " << argv[3] << endl;
    char *interf_name = argv[1];
    unsigned width = (unsigned) atoi(argv[2]);
    unsigned looks = (unsigned) atoi(argv[3]);
    int method = unsigned(atoi(argv[4]));
    string outfile = argv[5];
    vector<float> interf;
    interf = getVectorByFilename(interf_name,interf);
    if (interf.size() % width != 0) {
        cerr << "Unrecognized shape!\n";
    } else {
        unsigned length = (unsigned) interf.size() / width / 2;
        vector<float> interf_rel = interf;
        vector<float> interf_img = interf;
        interf_rel = retrieve_rel_int(interf_rel);
        interf_img = retrieve_img_int(interf_img);
        vector<vector<float> > rel_mat;
        rel_mat.resize(length, vector<float> (width,0.0));
        vector<vector<float> > img_mat;
        img_mat.resize(length, vector<float> (width,0.0));
        rel_mat = reshapeData(interf_rel,length,width,rel_mat);
        img_mat = reshapeData(interf_img,length,width,img_mat);
        unsigned new_length = (unsigned) length / looks;
        unsigned new_width = (unsigned) width / looks;
        int k = 0;
        vector<float> new_data(new_length*new_width*2,0.0);
        vector<float> temp_rel;
        vector<float> temp_img;
        for (int i=0; i<new_length*new_width; i++) {
            temp_rel.clear();
            temp_img.clear();
            float sum_rel = 0.0;
            float sum_img = 0.0;
            for (int k=0; k<looks; k++) {
                for (int l=0; l<looks; l++) {
                    if (fabs(rel_mat[i/new_width*looks+k][(i%new_width)*looks+l]) > 0.00001) {
                        temp_rel.push_back(rel_mat[i/new_width*looks+k][i%new_width*looks+l]);
                        temp_img.push_back(img_mat[i/new_width*looks+k][i%new_width*looks+l]);
                        sum_rel = sum_rel + rel_mat[i/new_width*looks+k][i%new_width*looks+l];
                        sum_img = sum_img + img_mat[i/new_width*looks+k][i%new_width*looks+l];
                        }
                    }
            }
            if (!temp_rel.empty()) {
                if (method) {
                    sort(temp_rel.begin(), temp_rel.end());
                    int size = temp_rel.size();
                    if (size  % 2 == 0) {
                        new_data[i*2] = (temp_rel[size/2-1] + temp_rel[size/2])/2;
                        new_data[i*2+1] = (temp_img[size/2-1] + temp_img[size/2])/2;
                    } else {
                        new_data[i*2] = temp_rel[size/2];
                        new_data[i*2+1] = temp_img[size/2];
                    }
                } else {
                    /*
                    new_data[i*2] = std::accumulate(temp_rel.begin(),temp_rel.begin(),0.0) / temp_rel.size();
                    new_data[i*2+1] = std::accumulate(temp_img.begin(),temp_img.begin(),0.0) / temp_img.size();
                    */
                    new_data[i*2] = sum_rel / temp_rel.size();
                    new_data[i*2+1] = sum_img / temp_img.size();
                    }
                }
        }
        if (fileExists(outfile)) { 
            cout << outfile << " exists! Removing!\n";
            string call_str = "rm -f ";
            call_str.append(outfile);
            char * call_char;
            call_char = &call_str[0];
            std::system(call_char);
        }
        ofstream File_temp;
        const char* pointer = reinterpret_cast<const char*>(&new_data[0]);
        File_temp.open(outfile, ios::out | ios::binary);
        File_temp.write(pointer, new_data.size()*4);
        File_temp.close();
    }
    return 0;
                        
}

