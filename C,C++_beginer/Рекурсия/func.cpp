int dvoich(int n)
{
	int k,c;
	k = n / 2;
	c = n % 2;
	if (k != 0)
	{
		return(dvoich(k) * 10 + c);
	}
	else return(1);
	

}