#include "SpreadsheetCell.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace std;

SpreadsheetCell::SpreadsheetCell()
: mValue(0)
{
}

SpreadsheetCell::SpreadsheetCell(double initialValue)
{
	set(initialValue);
}

SpreadsheetCell::SpreadsheetCell(const string& initialValue)
: mValue(stringToDouble(initialValue))
, mString(initialValue)
{
}

SpreadsheetCell::SpreadsheetCell(const SpreadsheetCell& src)
{
	mValue = src.mValue;
	mString = src.mString;
	mNumAccesses = src.mNumAccesses;
}

SpreadsheetCell& SpreadsheetCell::operator=(const SpreadsheetCell& rhs)
{
	if (this == &rhs) {
		return *this;
	}
	mValue = rhs.mValue;
	mString = rhs.mString;
	mNumAccesses = rhs.mNumAccesses;
	return *this;
}

void SpreadsheetCell::set(double inValue)
{
	mValue = inValue;
	mString = doubleToString(mValue);
}

void SpreadsheetCell::set(const string& inString)
{
	mString = inString;
	mValue = stringToDouble(mString);
}

string SpreadsheetCell::doubleToString(double inValue)
{
	ostringstream ostr;

	ostr << inValue;
	return ostr.str();
}

double SpreadsheetCell::stringToDouble(const string& inString)
{
	double temp;

	istringstream istr(inString);

	istr >> temp;
	if (istr.fail() || !istr.eof()) {
		return 0;
	}
	return temp;
}

SpreadsheetCell operator+(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	SpreadsheetCell newCell;
	newCell.set(lhs.mValue + rhs.mValue); // call set to update mValue and mString
	return newCell;
}

SpreadsheetCell operator-(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	SpreadsheetCell newCell;
	newCell.set(lhs.mValue - rhs.mValue); // call set to update mValue and mString
	return newCell;
}

SpreadsheetCell operator*(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	SpreadsheetCell newCell;
	newCell.set(lhs.mValue * rhs.mValue); // call set to update mValue and mString
	return newCell;
}

SpreadsheetCell operator/(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	if (rhs.mValue == 0)
		throw invalid_argument("Divide by zero.");
	SpreadsheetCell newCell;
	newCell.set(lhs.mValue / rhs.mValue); // call set to update mValue and mString
	return newCell;
}

SpreadsheetCell& SpreadsheetCell::operator+=(const SpreadsheetCell& rhs)
{
	set(mValue + rhs.mValue); // call set to update mValue and mString
	return *this;
}

SpreadsheetCell& SpreadsheetCell::operator-=(const SpreadsheetCell& rhs)
{
	set(mValue - rhs.mValue); // call set to update mValue and mString
	return *this;
}

SpreadsheetCell& SpreadsheetCell::operator*=(const SpreadsheetCell& rhs)
{
	set(mValue * rhs.mValue); // call set to update mValue and mString
	return *this;
}

SpreadsheetCell& SpreadsheetCell::operator/=(const SpreadsheetCell& rhs)
{
	if (rhs.mValue == 0)
		throw invalid_argument("Divide by zero.");
	set(mValue / rhs.mValue); // call set to update mValue and mString
	return *this;
}

bool operator==(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue == rhs.mValue;
}

bool operator<(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue < rhs.mValue;
}

bool operator>(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue > rhs.mValue;
}

bool operator!=(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue != rhs.mValue;
}

bool operator<=(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue <= rhs.mValue;
}

bool operator>=(const SpreadsheetCell& lhs, const SpreadsheetCell& rhs)
{
	return lhs.mValue >= rhs.mValue;
}

SpreadsheetCell SpreadsheetCell::operator-() const
{
	SpreadsheetCell newCell(*this);
	newCell.set(-mValue); // call set to update mValue and mString

	return newCell;
}

SpreadsheetCell& SpreadsheetCell::operator++()
{
	set(mValue + 1);
	return *this;
}

SpreadsheetCell SpreadsheetCell::operator++(int)
{
	SpreadsheetCell oldCell(*this); // save the current value before incrementing
	set(mValue + 1); // increment
	return oldCell;  // return the old value
}

SpreadsheetCell& SpreadsheetCell::operator--()
{
	set(mValue - 1);
	return *this;
}

SpreadsheetCell SpreadsheetCell::operator--(int)
{
	SpreadsheetCell oldCell(*this); // save the current value before decrementing
	set(mValue - 1); // decrement
	return oldCell;  // return the old value
}

ostream& operator<<(ostream& ostr, const SpreadsheetCell& cell)
{
	ostr << cell.mString;
	return ostr;
}

istream& operator>>(istream& istr, SpreadsheetCell& cell)
{
	string temp;
	istr >> temp;
	cell.set(temp);
	return istr;
}
