/*
 * FileTools.h
 *
 *  Created on: 2014年4月22日
 *      Author: darkblue
 */

#ifndef FILETOOLS_H
#define FILETOOLS_H

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>

using namespace std;

template <typename T>
class FileVectorTool {
  private:
  public:
    static vector<T> & getVectorByFilename(char *filename, vector<T> &v);
    static vector<vector<T> > & reshapeData(vector<T> &v, unsigned line, unsigned col, vector<vector<T> > &mat);
    static vector<T> & retrieve_rel_int(vector<T> &v);
    static vector<T> & retrieve_img_int(vector<T> &v);
    static vector<T> & RemoveOdd(vector<T>& v);
    static vector<T> & RemoveEven(vector<T>& v);
    bool is_odd(const int & number);
    bool is_even(const int & number);
    bool fileExists(const string & filename);
};


/**
 * @function
 * 此函数可以让用户通过二进制文件的名称获得一个向量
 *
 * @param
 * filename:一个char数组的首地址，本函数要求上不支持文件名中含有例如中文、日文等非西文字符
 * vector:你所需要的用来储存文件信息的矢量的引用
 * T:文件中存储的数据的类型
 *
 * @return
 * 与参数中vector地址相同的引用
 *
 * @exception
 * 当文件不存在、文件名字符非法、文件超过系统规定的大小，程序会发生未知的异常
 */
template<typename T>
vector<T> & getVectorByFilename(char *filename, vector<T> &v) {
	struct stat fileStat;
	stat(filename, &fileStat);//获取文件基本信息
	long filesize = fileStat.st_size;//确定文件的长度
	v.resize(filesize/sizeof(T));//给向量分配合适的内存空间
	ifstream file;
	file.open(filename, ifstream::in | ifstream::binary);
    
	//下面这段代码在处理大文件时，会对内存造成影响，影响程序后期性能
	file.read(reinterpret_cast<char*>(&v[0]), filesize);//读取文件到向量中
	file.close();//关闭文件流
	return v;
	/*
	 * 此函数只能在极其严格环境下使用，属于测试级代码
	 * 此函数有很多异常没有处理，比如空地址
	 * 内存和性能问题需要后期优化
	 */
}

template<typename T>
vector<vector<T> > & reshapeData(vector<T> &v, unsigned line, unsigned col, vector<vector<T> > & mat) {
    for (int i=0;i<line;i++) {
        for (int j=0;j<col;j++) {
            mat[i][j] = v[(i*col)+j];
        }
    }
    return mat;
}

bool is_odd(const int & number) { return ((number % 2) == 1);}

bool is_even(const int & number) { return ((number % 2) == 0);}

/*
template<typename T>
vector<T> & retrieve_img_int(vector<T> &v) {
    v.erase(remove_if(v.begin(), v.end(), is_even), v.end());
    return v;
}

template<typename T>
vector<T> & retrieve_rel_int(vector<T> &v) {
    v.erase(remove_if(v.begin(), v.end(), is_odd), v.end());
    cout << "New size is: " << v.size() << endl;
    return v;
}
*/

template<typename T>
vector<T> & retrieve_rel_int(vector<T> &v) {
    vector<T> new_v;
    for (int i=0;i<v.size();i++) {
        if (i%2==0) { new_v.push_back(v[i]); }
    }
    v.clear();
    v = new_v;
    new_v.clear();
    new_v.shrink_to_fit();
    return v;
}

template<typename T>
vector<T> & retrieve_img_int(vector<T> &v) {
    vector<T> new_v;
    for (int i=0;i<v.size();i++) {
        if (i%2==1) { new_v.push_back(v[i]); }
    }
    v.clear();
    v = new_v;
    new_v.clear();
    new_v.shrink_to_fit();
    return v;
}

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

#endif /* FILETOOLS_H */
