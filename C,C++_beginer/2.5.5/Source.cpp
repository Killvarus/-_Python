#include <iostream>
#include <windows.h>
#include "resource.h"
using namespace std;
/* ���� ���������� ���������� (������� ������� ������������ ���������)
 � ���� ��������: ���������� (������������ ����������) � ����������,
 ��������� � ������� �����, � ������� ������������ ��������� ����������.
*/
int x1, y, x2, y2;
int WINAPI DlgProc(HWND hDlg, WORD wMsg, WORD wParam, DWORD)
{
	PAINTSTRUCT ps;
	if (wMsg == WM_CLOSE || wMsg == WM_COMMAND && wParam == IDOK) {
		EndDialog(hDlg, 0);
	}
	else
		if (wMsg == WM_INITDIALOG) {
			/* ������ ������ ���� (��� OY ���������� ����): */
			RECT rc;
			GetClientRect(hDlg, &rc);
			int dx = rc.right - rc.left;
			int dy = rc.bottom - rc.top;
			/* ����� ����� ����� ���������������� ���������� ���������� ����������,
			 ����� ��� ���������� � ���� ������ ����. �� ����, ����� ����������
			 ��� ������ ����� ���������� ���������� ���������� � ������� ���������
			 ������� ����. ���� ���� ������������ ��� �� �����.
			*/
		}
		else
			if (wMsg == WM_PAINT) {
				BeginPaint(hDlg, &ps);
				/* ������� ���� �����: */
				HPEN hPen = (HPEN)CreatePen(PS_SOLID, 1, RGB(0, 0, 255));
				HPEN hOldPen = (HPEN)SelectObject(ps.hdc, hPen);
				/*
				 ��� ��� ������ ����� ����� ��������:
				 MoveToEx(ps.hdc,int x,int y,NULL);
				 ( ��������� �������� � ����� ��������� POINT
				 ��� �������� ��������� ���������� ������� )
				 LineTo(ps.hdc,int x,int y); (� ������� �������, �� ��� ����������)
				 TextOut(ps.hdc,int x,int y,char* szText,lstrlen(szText));
				 Rectangle(ps.hdc,int left,int top,int right,int bottom);
				 Ellipse(ps.hdc,int left,int top,int right,int bottom);
				 Polygon(ps.hdc,const POINT * lp,int nPoints);
				 Polyline(ps.hdc,const POINT * lp,int nPoints);
				 SetPixel(ps.hdc,int x,int y, RGB(red,green,blue));
				*/
				/* ����� � ������������ ��������� ����������� ���������� ���������
				 ���������� � ���� �������, � ����� �� ������ ���������� �����.
				*/
				RECT rc;
				GetClientRect(hDlg, &rc);
				cout << rc.right - rc.left << endl;
				cout << rc.bottom - rc.top<<endl;
				//cout << "Vvedite x0,y0" << endl;
				int x0, y0;
				//cin >> x0 >> y0;
				x0 = (rc.right - rc.left) / 2;
				y0 = (rc.bottom - rc.top) / 2;
				double Vx0,Vy0;
				cout << "Vvedite Vx0,Vy0"<<endl;
				cin >> Vx0 >> Vy0;
				double Vx,Vy;
				cout << "Vvedite B"<<endl;
				double B;
				cin >> B;
				cout << "Vvedite m"<<endl;
				double m;
				cin >> m;
				cout << "Vvedite q"<<endl;
				double q;
				cin >> q;
				const int k = 10000;
				int px[k];
				int py[k];
				px[0] = x0;
				py[0] = y0;
				int kol = 0;
				double dt=0.1;
				for (int i = 0;i<k; i++)
				{
					if (i != 0) {
						Vx0 = Vx;
						Vy0 = Vy;
						px[i] = px[i - 1] + Vx0 * dt;
						py[i] = py[i - 1] + Vy0 * dt;
					}
					else {
						px[i] = px[i] + Vx0 * dt;
						py[i] = py[i] + Vy0 * dt;
					}
					Vx = Vx0 + dt * Vy0 * B*q/m;
					Vy = Vy0 - dt * Vx0 * B*q/m;
					kol++;
					if (px[i] > (rc.right - rc.left) or px[i]<0 or py[i]> (rc.bottom - rc.top) or py[i] < 0)  break;
				}
				POINT* ptOld;
				ptOld = new POINT[kol];
				for (int i = 0; i < kol; i++) {
					ptOld[i].x = px[i];
					ptOld[i].y = py[i];
				}
				//MoveToEx(ps.hdc, x1, y, &ptOld);
				Polyline(ps.hdc, ptOld, kol);
				//LineTo(ps.hdc, x2, y2);
				//Rectangle(ps.hdc, x1, y,x2,y2 );
				/* ���� ��� ������ �� ���������, ��������� ���: */
				//SelectObject(ps.hdc, hOldPen);
				DeleteObject(hPen);
				EndPaint(hDlg, &ps);
			}
	return 0;
}
void main()
{
	

	/* ���� ���������� ������: */
//	cout << "Please, enter 4 coords:\n" << flush;
	//cin >> x1 >> y >> x2 >> y2;
	//cout << "x1 = " << x1 << "\ny1 = " << y
		//<< "\nx2 = " << x2 << "\ny2 = " << y2 << "\n" << flush;
	/*
 �����, ����� �������, ����� ��������� ����������
 ���� ����� ���������� � ���������� ������� ���������.
 ������� x-y ��������� ������ ���� �������� ��������� -
 � DlgProc � � ������� main.
 */
	DialogBox(NULL, MAKEINTRESOURCE(IDD_DIALOG1), NULL, (DLGPROC)DlgProc);
}
