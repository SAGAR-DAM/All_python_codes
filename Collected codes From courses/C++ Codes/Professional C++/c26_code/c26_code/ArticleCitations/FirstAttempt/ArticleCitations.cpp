#include "ArticleCitations.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
using namespace std;

ArticleCitations::ArticleCitations(const string& fileName)
{
	// All we have to do is read the file
	readFile(fileName);
}

ArticleCitations::ArticleCitations(const ArticleCitations& src)
{
	copy(src);
}

ArticleCitations& ArticleCitations::operator=(const ArticleCitations& rhs)
{
	// Check for self-assignment
	if (this == &rhs) {
		return *this;
	}
	// Free the old memory
	delete [] mCitations;
	// Copy the data
	copy(rhs);

	return *this;
}

void ArticleCitations::copy(const ArticleCitations& src)
{
	// Copy the article name, author, etc.
	mArticle = src.mArticle;
	// Copy the number of citations
	mNumCitations = src.mNumCitations;
	// Allocate an array of the correct size
	mCitations = new string[mNumCitations];
	// Copy each element of the array
	for (int i = 0; i < mNumCitations; i++) {
		mCitations[i] = src.mCitations[i];
	}
}

ArticleCitations::~ArticleCitations()
{
	delete [] mCitations;
}

void ArticleCitations::readFile(const string& fileName)
{
	// Open the file and check for failure.
	ifstream istr(fileName.c_str());
	if (istr.fail()) {
		throw invalid_argument("Unable to open file");
	}
	// Read the article author, title, etc. line.
	getline(istr, mArticle);

	// Skip the white space before the citations start.
	istr >> ws;

	int count = 0;
	// Save the current position so we can return to it.
	ios_base::streampos citationsStart = istr.tellg();
	// First count the number of citations.
	while (!istr.eof()) {
		// Skip white space before the next entry.
		istr >> ws;
		string temp;
		getline(istr, temp);
		if (!temp.empty()) {
			count++;
		}
	}

	if (count != 0) {
		// Allocate an array of strings to store the citations.
		mCitations = new string[count];
		mNumCitations = count;
		// Seek back to the start of the citations.
		istr.seekg(citationsStart);
		// Read each citation and store it in the new array.
		for (count = 0; count < mNumCitations; count++) {
			string temp;
			getline(istr, temp);
			if (!temp.empty()) {
				mCitations[count] = temp;
			}
		}
	} else {
		mNumCitations = -1;
	}
}
